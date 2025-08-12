# main.py
import modal
import os
from typing import Dict, TypedDict, Annotated
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
        "huggingface_hub==0.22.2",
        "hf-transfer==0.1.6",
        "torch==2.1.2",
        "langgraph==0.0.44",
        "numpy<2.0.0",
        "fastapi[standard]",
        "uvicorn",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# Create the Modal app
app = modal.App("dhya-persona-rag-pipeline", image=app_image)

# --- 2. Model Download and Management ---

@app.function(volumes={"/models": model_volume}, timeout=3600)
def download_models():
    """Download all required models to the persistent volume"""
    from huggingface_hub import snapshot_download
    import os
    
    print("Starting model downloads...")
    
    for model_name, model_id in MODEL_CONFIG.items():
        print(f"Downloading {model_name}: {model_id}")
        try:
            snapshot_download(
                repo_id=model_id,
                # Store under the repo id so agents can load with /models/<repo_id>
                local_dir=f"/models/{model_id}",
                local_dir_use_symlinks=False,
            )
            print(f"✅ Downloaded {model_name}")
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
        
        model_path = f"/models/{MODEL_CONFIG['orchestrator_model']}"
        engine_args = AsyncEngineArgs(
            model=model_path,
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

@app.cls(gpu="A10G", volumes={"/models": model_volume}, scaledown_window=300)
class RouterAgent:
    @modal.enter()
    def load_model(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        import time
        
        print("Loading router model...")
        start_time = time.time()
        
        model_path = f"/models/{MODEL_CONFIG['router_model']}"
        engine_args = AsyncEngineArgs(
            model=model_path,
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
        
        model_path = f"/models/{MODEL_CONFIG['retrieval_model']}"
        engine_args = AsyncEngineArgs(
            model=model_path,
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
        
        model_path = f"/models/{MODEL_CONFIG['synthesis_model']}"
        engine_args = AsyncEngineArgs(
            model=model_path,
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

            # For now, let's use a simplified approach that doesn't require the full workflow
            # This will help us test if the basic model loading works
            try:
                # Use the orchestrator directly for a simple response
                orchestrator = OrchestratorAgent()
                response = orchestrator.create_research_plan.remote(query, persona_context)
                
                return {"response": response}
            except Exception as workflow_error:
                return {"error": f"Model execution failed: {str(workflow_error)}"}
                
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
