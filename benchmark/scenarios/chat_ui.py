"""
AI Chat UI concurrency benchmark.

Tests how Open WebUI performs when multiple users are simultaneously
interacting with AI models through the actual browser UI using Playwright.
"""

import asyncio
import random
import time
from typing import Optional, List, Dict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from benchmark.core.base import BaseBenchmark
from benchmark.core.config import BenchmarkConfig
from benchmark.core.metrics import MetricsCollector, BenchmarkResult
from benchmark.clients.http_client import OpenWebUIClient, ClientPool
from benchmark.clients.browser_client import BrowserClient, BrowserPool
from benchmark.auth import ensure_admin_authenticated, AuthenticationError, ServiceNotReadyError


console = Console()


class ChatUIBenchmark(BaseBenchmark):
    """
    Benchmark for testing concurrent AI chat performance via browser UI.
    
    This benchmark:
    1. Verifies the configured model is available
    2. Creates test users for concurrent chat sessions
    3. Launches browser instances/contexts for each user
    4. Sends concurrent chat requests through the actual UI
    5. Measures TTFT, tokens/second, and total response time as rendered in browser
    """
    
    name = "Chat UI Concurrency"
    description = "Test concurrent AI chat performance via browser UI"
    version = "1.0.0"
    
    def __init__(
        self,
        config: BenchmarkConfig,
        admin_client: Optional[OpenWebUIClient] = None,
    ):
        """
        Initialize the chat UI concurrency benchmark.
        
        Args:
            config: Benchmark configuration
            admin_client: Optional pre-authenticated admin client
        """
        super().__init__(config)
        
        self._admin_client: Optional[OpenWebUIClient] = admin_client
        self._owns_admin_client: bool = admin_client is None
        self._http_client_pool: Optional[ClientPool] = None
        self._browser_pool: Optional[BrowserPool] = None
        self._test_clients: List[OpenWebUIClient] = []
        self._user_credentials: List[Dict[str, str]] = []
    
    async def setup(self) -> None:
        """
        Set up the benchmark environment.
        
        Authenticates admin, verifies model availability, configures model
        for public access, creates test users, and launches browsers.
        """
        # Authenticate admin if needed
        if self._admin_client is None:
            try:
                self._admin_client, _auth_result = await ensure_admin_authenticated(
                    base_url=self.config.target_url,
                    timeout=self.config.request_timeout,
                    wait_for_service=True,
                )
            except ServiceNotReadyError as e:
                raise RuntimeError(f"Open WebUI service not ready: {e}")
            except AuthenticationError as e:
                raise RuntimeError(f"Admin authentication failed: {e}")
        
        # Verify the configured model is available
        chat_config = self.config.chat
        model_available = await self._admin_client.verify_model_available(chat_config.model)
        
        if not model_available:
            available_models = await self._admin_client.get_models()
            model_names = [m.get("id", m.get("name", str(m))) for m in available_models[:10]]
            raise RuntimeError(
                f"Model '{chat_config.model}' is not available. "
                f"Available models: {', '.join(model_names)}"
                f"\nEnsure OpenAI API key is configured in Open WebUI settings, "
                f"or set CHAT_BENCHMARK_MODEL to an available model."
            )
        
        # Configure model for public access so benchmark users can use it
        made_public = await self._admin_client.make_model_public(chat_config.model)
        if not made_public:
            raise RuntimeError(
                f"Failed to configure model '{chat_config.model}' for public access. "
                f"Ensure you have admin privileges."
            )
        
        # Initialize HTTP client pool and create test users
        self._http_client_pool = ClientPool(
            self.config.target_url,
            self.config.request_timeout,
        )
        
        user_count = chat_config.max_concurrent_users
        
        # Create benchmark users with progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(f"Creating {user_count} users...", total=user_count * 2)
            
            def update_progress(current: int, total: int):
                progress.update(task, completed=current)
            
            self._test_clients = await self._http_client_pool.create_benchmark_users(
                admin_client=self._admin_client,
                count=user_count,
                email_pattern="chat_ui_benchmark_{n}@test.local",
                password="chat_ui_benchmark_pass_123",
                name_pattern="Chat UI Benchmark User {n}",
                progress_callback=update_progress,
            )
        
        # Store credentials for browser login (note: email pattern uses {n} = 1-indexed)
        self._user_credentials = [
            {
                "email": f"chat_ui_benchmark_{i + 1}@test.local",
                "password": "chat_ui_benchmark_pass_123",
            }
            for i in range(user_count)
        ]
        
        console.print(f"[green]✓[/green] Created {len(self._test_clients)} users")
        
        # Initialize browser pool
        browser_config = self.config.browser
        self._browser_pool = BrowserPool(
            base_url=self.config.target_url,
            headless=browser_config.headless,
            slow_mo=browser_config.slow_mo,
            viewport_width=browser_config.viewport_width,
            viewport_height=browser_config.viewport_height,
            timeout=browser_config.browser_timeout,
            use_isolated_browsers=browser_config.use_isolated_browsers,
        )
        
        # Launch browsers and login users
        console.print("[dim]Launching browsers and logging in users...[/dim]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Initializing browser pool...", total=user_count + 1)
            
            await self._browser_pool.initialize()
            progress.update(task, completed=1, description="Logging in users...")
            
            def login_progress(current: int, total: int):
                progress.update(task, completed=1 + current, description=f"Logged in {current}/{total} users...")
            
            # Create and login browser clients in batches
            browser_clients = await self._browser_pool.create_clients(
                credentials=self._user_credentials,
                login=True,
                batch_size=10,  # Login 10 users at a time
                progress_callback=login_progress,
            )
            progress.update(task, completed=user_count + 1)
        
        console.print(f"[green]✓[/green] Launched {len(browser_clients)} browser sessions")
    
    async def run(self) -> BenchmarkResult:
        """
        Execute the chat UI concurrency benchmark.
        
        Runs concurrent browser chat requests and measures performance metrics.
        
        Returns:
            BenchmarkResult with timing and token metrics
        """
        chat_config = self.config.chat
        metrics = MetricsCollector()
        metrics.start()
        
        browser_clients = self._browser_pool.clients
        user_count = len(browser_clients)
        requests_per_user = chat_config.requests_per_user
        total_requests = user_count * requests_per_user
        
        console.print(f"\n[bold cyan]Running chat UI benchmark[/bold cyan]")
        console.print(f"[dim]Model: {chat_config.model}[/dim]")
        console.print(f"[dim]Concurrent browser sessions: {user_count}[/dim]")
        console.print(f"[dim]Requests per user: {requests_per_user}[/dim]")
        console.print(f"[dim]Total requests: {total_requests}[/dim]")
        console.print(f"[dim]Headless: {self.config.browser.headless}[/dim]\n")
        
        # Track completed requests
        completed = 0
        errors = 0
        
        # Progress bar for tracking requests
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        )
        progress.start()
        task = progress.add_task("Running requests...", total=total_requests)
        
        def update_status():
            progress.update(task, completed=completed + errors, 
                          description=f"Completed: {completed} | Errors: {errors}")
        
        async def run_user_session(client: BrowserClient, user_num: int):
            """Run chat requests for a single browser user."""
            nonlocal completed, errors
            
            # Note: Model selection removed - it interferes with the chat input
            # The default model should work, or model can be pre-selected in Open WebUI
            
            for req_num in range(requests_per_user):
                # Select a random prompt from the pool
                user_prompt = random.choice(chat_config.prompt_pool)
                
                try:
                    # Start a new chat for clean state (optional but cleaner)
                    if req_num > 0:
                        await client.start_new_chat()
                    
                    result = await client.send_message_and_wait(
                        message=user_prompt,
                        timeout_ms=self.config.browser.browser_timeout,
                    )
                    
                    if result.success:
                        metrics.record_streaming_timing(
                            operation="chat_completion_ui",
                            duration_ms=result.total_duration_ms,
                            ttft_ms=result.ttft_ms,
                            tokens_generated=result.tokens_rendered,
                            success=True,
                            metadata={
                                "user": user_num,
                                "request": req_num,
                                "model": chat_config.model,
                                "prompt": user_prompt[:50],
                                "content_preview": result.content[:100] if result.content else "",
                            },
                        )
                        completed += 1
                    else:
                        metrics.record_streaming_timing(
                            operation="chat_completion_ui",
                            duration_ms=result.total_duration_ms,
                            ttft_ms=0,
                            tokens_generated=0,
                            success=False,
                            error=result.error or "Unknown error",
                            metadata={
                                "user": user_num,
                                "request": req_num,
                                "model": chat_config.model,
                            },
                        )
                        errors += 1
                    
                    update_status()
                    
                except Exception as e:
                    metrics.record_streaming_timing(
                        operation="chat_completion_ui",
                        duration_ms=0,
                        ttft_ms=0,
                        tokens_generated=0,
                        success=False,
                        error=str(e),
                        metadata={
                            "user": user_num,
                            "request": req_num,
                            "model": chat_config.model,
                        },
                    )
                    errors += 1
                    update_status()
                
                # Small delay between requests from same user
                await asyncio.sleep(0.5)
        
        # Run all user sessions concurrently
        tasks = [
            run_user_session(client, i) 
            for i, client in enumerate(browser_clients)
        ]
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.stop()
        
        # Finish progress bar
        progress.update(task, description="Complete!", completed=total_requests)
        progress.stop()
        
        return metrics.get_result(
            benchmark_name=self.name,
            concurrent_users=user_count,
            metadata={
                "model": chat_config.model,
                "requests_per_user": requests_per_user,
                "total_requests": total_requests,
                "prompt_pool_size": len(chat_config.prompt_pool),
                "headless": self.config.browser.headless,
                "browser_timeout_ms": self.config.browser.browser_timeout,
            },
        )
    
    async def teardown(self) -> None:
        """Clean up benchmark resources."""
        # Close browser pool
        if self._browser_pool:
            await self._browser_pool.close_all()
        
        # Close HTTP test user clients and clean up users
        if self._http_client_pool:
            await self._http_client_pool.close_all()
        
        # Close admin client if we created it
        if self._admin_client and self._owns_admin_client:
            await self._admin_client.close()
