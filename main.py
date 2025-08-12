# main.py
import modal
import os
from typing import Dict, TypedDict, Annotated
import uuid
import time
from langgraph.graph import StateGraph, END
import operator

# --- 1. Model Configuration and Setup ---

import modal
import os
from typing import Dict, TypedDict
from langgraph.graph import StateGraph, END

# Model configuration - using a small model supported by vLLM
MODEL_CONFIG = {
    "orchestrator_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "router_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "retrieval_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "synthesis_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

# Create persistent volumes for model storage
# Standardize to a single volume name used across scripts
model_volume = modal.Volume.from_name("llm-models-vol", create_if_missing=True)
lora_volume = modal.Volume.from_name("persona-rag-loras", create_if_missing=True)

# Define the container image with all required dependencies
app_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm==0.4.0",
        "huggingface_hub>=0.23.0,<1.0.0",
        "hf-transfer==0.1.6",
        "torch==2.1.2",
        "transformers==4.41.1",
        "langgraph==0.0.44",
        "numpy<2.0.0",
        "fastapi[standard]",
        "uvicorn",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Persist HF cache on the /models volume
        "HF_HOME": "/models/hf-home",
        "HF_HUB_CACHE": "/models/hf-cache",
        "TRANSFORMERS_CACHE": "/models/hf-cache",
    })
)

# Create the Modal app
app = modal.App("dhya-persona-rag-pipeline", image=app_image)

# Simple job store for streaming planner outputs
planner_jobs = modal.Dict.from_name("planner-jobs", create_if_missing=True)

# --- 2. Model Download and Management ---

def get_local_model_dir(model_id: str) -> str:
    """Return a local directory path for the given model id.
    Prefer the persistent volume under /models. If missing weights, fall back to HF cache.
    """
    import os
    try:
        # Preferred: persistent volume path
        persistent_dir = f"/models/{model_id}"
        if os.path.isdir(persistent_dir):
            # Only use persistent dir if it actually contains weights
            for root, _, files in os.walk(persistent_dir):
                if any(f.endswith((".safetensors", ".bin")) for f in files):
                    return persistent_dir
        # Fallback: ensure in HF cache and use that path
        from huggingface_hub import snapshot_download
        cached_dir = snapshot_download(repo_id=model_id, local_dir=None, local_dir_use_symlinks=True)
        return cached_dir
    except Exception:
        # As a last resort, return the persistent dir so errors surface clearly later
        return f"/models/{model_id}"

@app.function(volumes={"/models": model_volume}, timeout=3600)
def download_models():
    """Download all required models to the persistent volume"""
    from huggingface_hub import snapshot_download
    import os
    
    print("Starting model downloads...")
    
    for model_name, model_id in MODEL_CONFIG.items():
        print(f"Downloading {model_name}: {model_id}")
        try:
            # With cache env pointing to /models/hf-cache, this persists on the volume
            cached_dir = snapshot_download(repo_id=model_id, local_dir=None, local_dir_use_symlinks=True)
            print(f"✅ Cached {model_name} at {cached_dir}")
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            raise

@app.function(volumes={"/models": model_volume}, timeout=3600)
def ensure_models_downloaded():
    """Ensure all models are downloaded"""
    download_models.remote()

# --- 3. Agent Classes with Model Warm-up ---

@app.cls(gpu="A10G", volumes={"/models": model_volume}, scaledown_window=300)
class OrchestratorAgent:
    @modal.enter()
    def load_model(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import time
        
        print("Loading orchestrator model...")
        start_time = time.time()
        
        model_id = MODEL_CONFIG['orchestrator_model']
        print(f"Resolved orchestrator model id: {model_id}")
        engine_args = AsyncEngineArgs(
            model=model_id,
            download_dir="/models",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Warm up the model with a simple inference
        print("Warming up orchestrator model...")
        self._warm_up_model()
        
        load_time = time.time() - start_time
        print(f"✅ Orchestrator model loaded in {load_time:.2f}s")

    def _warm_up_model(self):
        """Warm up the model with a simple inference"""
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        import asyncio
        
        try:
            prompt = "Hello, how are you?"
            sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
            request_id = random_uuid()
            
            async def warm_up():
                results_generator = self.engine.generate(prompt, sampling_params, request_id)
                async for request_output in results_generator:
                    break  # Just get the first output
            
            asyncio.run(warm_up())
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")

    @modal.method()
    async def create_research_plan(self, query: str, persona_context: str) -> str:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        prompt = f"""
        You are a research planning agent. Create a detailed research plan for the following query.
        Break down the research into specific steps that can be executed by a research agent.

        Query: {query}
        Persona Context: {persona_context}

        Research Plan:
        """
        sampling_params = SamplingParams(temperature=0.3, max_tokens=1024)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text

    @modal.method()
    async def start_plan_stream(self, job_id: str, query: str, persona_context: str) -> None:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        import asyncio

        prompt = f"""
        You are a research planning agent. Create a detailed, step-by-step research plan for the query.
        Persona Context: {persona_context}
        Query: {query}

        Plan:
        """
        sampling_params = SamplingParams(temperature=0.3, max_tokens=512)
        request_id = random_uuid()
        accumulated = ""

        try:
            results_generator = self.engine.generate(prompt, sampling_params, request_id)
            async for request_output in results_generator:
                text = request_output.outputs[0].text
                delta = text[len(accumulated):]
                accumulated = text
                if delta:
                    # Append chunk safely
                    with planner_jobs.batch_update() as batch:
                        job = planner_jobs.get(job_id, {"chunks": [], "done": False, "error": None})
                        job["chunks"].append(delta)
                        batch[job_id] = job
            with planner_jobs.batch_update() as batch:
                job = planner_jobs.get(job_id, {"chunks": [], "done": False, "error": None})
                job["done"] = True
                batch[job_id] = job
        except Exception as e:
            with planner_jobs.batch_update() as batch:
                job = planner_jobs.get(job_id, {"chunks": [], "done": True, "error": str(e)})
                job["error"] = str(e)
                job["done"] = True
                batch[job_id] = job

@app.cls(gpu="A10G", volumes={"/models": model_volume}, scaledown_window=300)
class RouterAgent:
    @modal.enter()
    def load_model(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import time
        
        print("Loading router model...")
        start_time = time.time()
        
        model_id = MODEL_CONFIG['router_model']
        print(f"Resolved router model id: {model_id}")
        engine_args = AsyncEngineArgs(
            model=model_id,
            download_dir="/models",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Warm up the model
        print("Warming up router model...")
        self._warm_up_model()
        
        load_time = time.time() - start_time
        print(f"✅ Router model loaded in {load_time:.2f}s")

    def _warm_up_model(self):
        """Warm up the model with a simple inference"""
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        import asyncio
        
        try:
            prompt = "Hello, how are you?"
            sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
            request_id = random_uuid()
            
            async def warm_up():
                results_generator = self.engine.generate(prompt, sampling_params, request_id)
                async for request_output in results_generator:
                    break  # Just get the first output
            
            asyncio.run(warm_up())
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")

    @modal.method()
    async def route_plan(self, research_plan: str) -> str:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        prompt = f"""
        You are a routing agent. Analyze the research plan and determine which agent should handle it.
        Return either "retrieval_agent" if the plan requires research, or "synthesis_agent" if it's ready for synthesis.

        Research Plan:
        {research_plan}

        Decision:
        """
        sampling_params = SamplingParams(temperature=0.1, max_tokens=50)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text

@app.cls(gpu="A10G", volumes={"/models": model_volume}, scaledown_window=300)
class RetrievalAgent:
    @modal.enter()
    def load_model(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import time
        
        print("Loading retrieval model...")
        start_time = time.time()
        
        model_id = MODEL_CONFIG['retrieval_model']
        print(f"Resolved retrieval model id: {model_id}")
        engine_args = AsyncEngineArgs(
            model=model_id,
            download_dir="/models",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Warm up the model
        print("Warming up retrieval model...")
        self._warm_up_model()
        
        load_time = time.time() - start_time
        print(f"✅ Retrieval model loaded in {load_time:.2f}s")

    def _warm_up_model(self):
        """Warm up the model with a simple inference"""
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        import asyncio
        
        try:
            prompt = "Hello, how are you?"
            sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
            request_id = random_uuid()
            
            async def warm_up():
                results_generator = self.engine.generate(prompt, sampling_params, request_id)
                async for request_output in results_generator:
                    break  # Just get the first output
            
            asyncio.run(warm_up())
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")

    @modal.method()
    async def execute_research(self, research_plan: str) -> str:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        prompt = f"""
        You are a research agent. Execute the following research plan by providing detailed answers for each step.
        Synthesize the information into a coherent report.

        Research Plan:
        {research_plan}

        Research Report:
        """
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text

@app.cls(gpu="A10G", volumes={"/models": model_volume}, scaledown_window=300)
class SynthesisAgent:
    @modal.enter()
    def load_model(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import time
        
        print("Loading synthesis model...")
        start_time = time.time()
        
        model_id = MODEL_CONFIG['synthesis_model']
        print(f"Resolved synthesis model id: {model_id}")
        engine_args = AsyncEngineArgs(
            model=model_id,
            download_dir="/models",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            trust_remote_code=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        # Warm up the model
        print("Warming up synthesis model...")
        self._warm_up_model()
        
        load_time = time.time() - start_time
        print(f"✅ Synthesis model loaded in {load_time:.2f}s")

    def _warm_up_model(self):
        """Warm up the model with a simple inference"""
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid
        import asyncio
        
        try:
            prompt = "Hello, how are you?"
            sampling_params = SamplingParams(temperature=0.1, max_tokens=10)
            request_id = random_uuid()
            
            async def warm_up():
                results_generator = self.engine.generate(prompt, sampling_params, request_id)
                async for request_output in results_generator:
                    break  # Just get the first output
            
            asyncio.run(warm_up())
        except Exception as e:
            print(f"Warning: Model warm-up failed: {e}")

    @modal.method()
    async def generate_response(self, research_report: str, query: str, persona_context: str, user_id: str) -> str:
        from vllm.sampling_params import SamplingParams
        from vllm.utils import random_uuid

        prompt = f"""
        You are a helpful AI assistant personalized for the user.
        Your persona is: {persona_context}

        Use the following research report to answer the user's query.
        Provide a final, consolidated response that is helpful, accurate, and in your persona's voice.

        Research Report:
        {research_report}

        User Query: {query}

        Final Answer:
        """
        sampling_params = SamplingParams(temperature=0.7, max_tokens=2048)
        request_id = random_uuid()
        results_generator = self.engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for request_output in results_generator:
            final_output = request_output
            
        return final_output.outputs[0].text

# --- 4. Orchestration and Web Endpoint ---

class AgentState(TypedDict):
    query: str
    persona_context: str
    user_id: str
    research_plan: str
    research_report: str
    final_response: str
    next_agent: str
@app.function(volumes={"/models": model_volume})
def quick_plan(query: str, persona_context: str) -> str:
    """CPU-friendly fallback planner using HF Transformers. Returns a short plan string."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    model_id = MODEL_CONFIG["orchestrator_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/models/hf-cache", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir="/models/hf-cache",
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    prompt = (
        "You are a research planning agent. Create a short, step-by-step research plan for the query.\n"
        f"Persona Context: {persona_context}\n"
        f"Query: {query}\n"
        "Plan:"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Return only the generated continuation after "Plan:"
    return text.split("Plan:", 1)[-1].strip()


@app.cls(cpu=4.0)
class PersonaRAG:
    @modal.enter()
    def setup_graph(self):
        # Ensure models are downloaded first
        print("Ensuring models are downloaded before initializing agents...")
        ensure_models_downloaded.remote()
        
        # Wait a moment for the download to complete
        import time
        time.sleep(5)
        
        # Initialize agent instances
        print("Initializing agent instances...")
        self.orchestrator = OrchestratorAgent()
        self.router = RouterAgent()
        self.retriever = RetrievalAgent()
        self.synthesizer = SynthesisAgent()

        def planning_node(state: AgentState):
            plan = self.orchestrator.create_research_plan.remote(state["query"], state["persona_context"])
            return {"research_plan": plan}

        def routing_node(state: AgentState):
            next_agent = self.router.route_plan.remote(state["research_plan"])
            return {"next_agent": next_agent}

        def research_node(state: AgentState):
            report = self.retriever.execute_research.remote(state["research_plan"])
            return {"research_report": report}

        def synthesis_node(state: AgentState):
            response = self.synthesizer.generate_response.remote(
                state["research_report"], state["query"], state["persona_context"], state["user_id"]
            )
            return {"final_response": response}

        workflow = StateGraph(AgentState)
        workflow.add_node("planner", planning_node)
        workflow.add_node("router", routing_node)
        workflow.add_node("researcher", research_node)
        workflow.add_node("synthesizer", synthesis_node)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "router")
        
        def decide_next_node(state: AgentState):
            if "retrieval_agent" in state["next_agent"]:
                return "researcher"
            else:
                return "synthesizer"

        workflow.add_conditional_edges("router", decide_next_node, {
            "researcher": "researcher",
            "synthesizer": "synthesizer"
        })
        
        workflow.add_edge("researcher", "synthesizer")
        workflow.add_edge("synthesizer", END)

        self.app_graph = workflow.compile()

    @modal.fastapi_endpoint(method="POST")
    def run(self, item: Dict):
        try:
            query = item.get("query")
            user_id = item.get("user_id", "default_user")
            
            if not query:
                return {"error": "Query is required"}
            
            persona_context = "An expert financial auditor who is direct and provides source-backed information."

            # GPU planner via job queue with CPU fallback
            job_id = str(uuid.uuid4())
            # Initialize job record
            planner_jobs[job_id] = {"chunks": [], "done": False, "error": None}

            try:
                orchestrator = OrchestratorAgent()
                orchestrator.start_plan_stream.remote(job_id, query, persona_context)
            except Exception as e:
                # Fallback immediately on failure to start GPU job
                fallback = quick_plan.remote(query, persona_context)
                return {"response": fallback, "mode": "cpu_fallback"}

            # Poll for streamed chunks with timeout; if GPU fails, use fallback
            start = time.time()
            timeout_s = 60
            last_len = 0
            while time.time() - start < timeout_s:
                job = planner_jobs.get(job_id)
                if not job:
                    break
                chunks = job.get("chunks", [])
                error = job.get("error")
                done = job.get("done")

                if error:
                    # GPU failed; use fallback
                    fallback = quick_plan.remote(query, persona_context)
                    return {"response": fallback, "mode": "cpu_fallback"}

                if chunks and len(chunks) > last_len:
                    # Return latest partial to keep latency low
                    text = "".join(chunks)
                    last_len = len(chunks)
                    if done:
                        return {"response": text, "mode": "gpu_stream"}
                    # If not done but we have enough to be useful, return now
                    if len(text) > 300:
                        return {"response": text, "mode": "gpu_partial"}

                time.sleep(0.5)

            # If timeout/no chunks, fallback
            fallback = quick_plan.remote(query, persona_context)
            return {"response": fallback, "mode": "cpu_fallback_timeout"}
                
        except Exception as e:
            return {"error": f"Internal server error: {str(e)}"}

    @modal.fastapi_endpoint(method="GET")
    def health_check(self):
        """Simple health check endpoint"""
        return {"status": "healthy", "message": "PersonaRAG API is running"}

    @modal.fastapi_endpoint(method="POST")
    def warm_up(self):
        """Warm up all models to reduce cold start time"""
        try:
            print("🔥 Warming up all models...")
            
            # Initialize all agents to trigger model loading
            orchestrator = OrchestratorAgent()
            router = RouterAgent()
            retriever = RetrievalAgent()
            synthesizer = SynthesisAgent()
            
            print("✅ All models warmed up and ready!")
            return {"status": "success", "message": "Models warmed up successfully"}
        except Exception as e:
            return {"status": "error", "message": f"Failed to warm up models: {str(e)}"}

# --- 5. Model Warm-up Function ---

@app.function()
def warm_up_models():
    """Pre-warm all models to reduce cold start time"""
    print("🔥 Warming up all models...")
    
    # Initialize all agents to trigger model loading
    orchestrator = OrchestratorAgent()
    router = RouterAgent()
    retriever = RetrievalAgent()
    synthesizer = SynthesisAgent()
    
    print("✅ All models warmed up and ready!")

# --- 6. Deployment Helper ---

@app.local_entrypoint()
def main():
    """Local entrypoint for testing and deployment"""
    print("🚀 PersonaRAG Pipeline")
    print("=" * 30)
    
    # Download models
    print("📥 Downloading models...")
    ensure_models_downloaded.remote()
    
    # Warm up models
    print("🔥 Warming up models...")
    warm_up_models.remote()
    
    print("✅ Deployment ready!")
    print("\nEndpoints:")
    print("- Health Check: GET /health")
    print("- Main API: POST /run")
