"""
Browser client for UI-based benchmarking using Playwright.

Provides async browser automation for testing Open WebUI's actual UI performance,
including login, chat interactions, and streaming response rendering.
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright


@dataclass
class BrowserChatResult:
    """Result from a browser-based chat interaction."""
    content: str
    ttft_ms: float  # Time to first token appearing in UI
    total_duration_ms: float  # Time until streaming complete
    tokens_rendered: int  # Approximate tokens based on content length
    success: bool
    error: Optional[str] = None
    
    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second based on render time."""
        if self.total_duration_ms > 0 and self.tokens_rendered > 0:
            return self.tokens_rendered / (self.total_duration_ms / 1000)
        return 0.0


class BrowserClient:
    """
    Async browser client for Open WebUI UI interactions.
    
    Wraps Playwright to provide high-level methods for login,
    chat interactions, and performance measurement.
    """
    
    # CSS Selectors for Open WebUI elements
    # These may need updates as Open WebUI's UI evolves
    SELECTORS = {
        # Login page - use simplest selectors that work
        "email_input": 'input[type="email"]',
        "password_input": 'input[type="password"]',
        "login_button": 'button[type="submit"]',
        
        # Main chat interface - Open WebUI uses contenteditable div for chat input
        "chat_input": '[contenteditable="true"]',
        "send_button": 'button[type="submit"]:has(svg), button[aria-label*="Send"]',
        "new_chat_button": 'button:has-text("New Chat"), a[href="/"]',
        
        # Model selector
        "model_selector": 'button[aria-label*="Model"], .model-selector, [data-testid="model-selector"]',
        "model_option": 'div[role="option"], button[role="menuitem"]',
        
        # Response area - Open WebUI uses message IDs
        "message_container": '[id^="message-"]',  # All messages have id="message-{uuid}"
        "response_prose": '.prose',  # Response content is in .prose elements
        "streaming_indicator": '.typing-indicator, [class*="loading"], [class*="streaming"]',
    }
    
    def __init__(
        self,
        base_url: str,
        headless: bool = True,
        slow_mo: int = 0,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: float = 30000,
    ):
        """
        Initialize the browser client.
        
        Args:
            base_url: Base URL of the Open WebUI instance
            headless: Run browser in headless mode
            slow_mo: Slow down operations by ms (for debugging)
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            timeout: Default timeout in milliseconds
        """
        self.base_url = base_url.rstrip('/')
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._is_logged_in: bool = False
    
    async def launch(self) -> None:
        """Launch the browser and create a new context."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo,
        )
        self._context = await self._browser.new_context(
            viewport={"width": self.viewport_width, "height": self.viewport_height},
        )
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout)
    
    async def close(self) -> None:
        """Close the browser and clean up resources."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None
        self._is_logged_in = False
    
    @property
    def page(self) -> Page:
        """Get the current page, raising if not initialized."""
        if self._page is None:
            raise RuntimeError("Browser not launched. Call launch() first.")
        return self._page
    
    @property
    def is_logged_in(self) -> bool:
        """Check if the client is logged in."""
        return self._is_logged_in
    
    async def login(self, email: str, password: str, max_retries: int = 3) -> bool:
        """
        Log in to Open WebUI with retry logic.
        
        Args:
            email: User email
            password: User password
            max_retries: Maximum number of login attempts
            
        Returns:
            True if login successful
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return await self._attempt_login(email, password)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    # Wait before retry with exponential backoff
                    await asyncio.sleep(1 * (attempt + 1))
        
        raise RuntimeError(f"Login failed after {max_retries} attempts: {last_error}")
    
    async def _attempt_login(self, email: str, password: str) -> bool:
        """Single login attempt."""
        try:
            # Navigate to login page
            await self.page.goto(f"{self.base_url}/auth", wait_until="networkidle")
            
            # Wait for login form to be ready
            await self.page.wait_for_selector(
                self.SELECTORS["email_input"],
                state="visible",
                timeout=self.timeout,
            )
            
            # Small delay to ensure form is interactive
            await self.page.wait_for_timeout(200)
            
            # Fill in credentials
            await self.page.fill(self.SELECTORS["email_input"], email)
            await self.page.fill(self.SELECTORS["password_input"], password)
            
            # Click login button
            await self.page.click(self.SELECTORS["login_button"])
            
            # Wait for navigation - give more time under load
            try:
                await self.page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass  # Continue to check URL
            
            # Check if we're no longer on /auth page (allow time for redirect)
            for _ in range(10):  # Check up to 10 times over 5 seconds
                current_url = self.page.url
                if "/auth" not in current_url:
                    # Look for chat input as indicator of successful login
                    try:
                        await self.page.wait_for_selector(
                            self.SELECTORS["chat_input"],
                            state="visible",
                            timeout=5000,
                        )
                    except Exception:
                        pass  # Chat input not found but we're past auth
                    self._is_logged_in = True
                    return True
                await self.page.wait_for_timeout(500)
            
            raise RuntimeError("Login did not redirect from auth page")
                
        except Exception as e:
            self._is_logged_in = False
            raise RuntimeError(f"Login failed: {e}")
    
    async def select_model(self, model_id: str) -> bool:
        """
        Select a specific model in the UI.
        
        Args:
            model_id: Model identifier to select
            
        Returns:
            True if model was selected successfully
        """
        try:
            # Click model selector dropdown
            model_selector = await self.page.wait_for_selector(
                self.SELECTORS["model_selector"],
                state="visible",
                timeout=5000,
            )
            if model_selector:
                await model_selector.click()
                
                # Wait for dropdown and find the model option
                await self.page.wait_for_timeout(500)  # Allow dropdown to render
                
                # Try to find and click the model by text content
                model_option = self.page.get_by_text(model_id, exact=False)
                if await model_option.count() > 0:
                    await model_option.first.click()
                    return True
                    
            return False
        except Exception:
            # Model selection is optional - may already be selected
            return False
    
    async def start_new_chat(self) -> bool:
        """
        Start a new chat session.
        
        Returns:
            True if new chat started successfully
        """
        try:
            new_chat_btn = await self.page.wait_for_selector(
                self.SELECTORS["new_chat_button"],
                state="visible",
                timeout=5000,
            )
            if new_chat_btn:
                await new_chat_btn.click()
                await self.page.wait_for_timeout(500)
                return True
            return False
        except Exception:
            return False
    
    async def send_message_and_wait(
        self,
        message: str,
        timeout_ms: float = 60000,
    ) -> BrowserChatResult:
        """
        Send a chat message and wait for the streaming response to complete.
        
        Measures time-to-first-token and total response time as seen in the UI.
        
        Args:
            message: The message to send
            timeout_ms: Maximum time to wait for response
            
        Returns:
            BrowserChatResult with timing metrics and content
        """
        start_time = time.time()
        first_token_time: Optional[float] = None
        
        try:
            # Find the chat input (contenteditable element)
            chat_input = await self.page.wait_for_selector(
                self.SELECTORS["chat_input"],
                state="visible",
                timeout=10000,
            )
            
            if not chat_input:
                return BrowserChatResult(
                    content="",
                    ttft_ms=0,
                    total_duration_ms=0,
                    tokens_rendered=0,
                    success=False,
                    error="Chat input not found",
                )
            
            # For contenteditable, we need to click and type (fill doesn't work)
            await chat_input.click()
            await chat_input.press("Control+a")  # Select all existing content
            await self.page.keyboard.type(message)
            
            # Count initial messages (including user + assistant messages)
            initial_messages = await self.page.query_selector_all(self.SELECTORS["message_container"])
            initial_count = len(initial_messages)
            
            # Send the message (press Enter)
            await self.page.keyboard.press("Enter")
            send_time = time.time()
            
            # Wait for new messages to appear (user message + assistant response)
            # Expect: initial_count + 2 (user's message + assistant's response)
            target_message_count = initial_count + 2
            
            content = ""
            last_content_length = 0
            stable_count = 0
            max_stable_checks = 5  # Content stable for 5 checks = 500ms
            response_element = None
            found_response = False
            
            while (time.time() - send_time) * 1000 < timeout_ms:
                # Check for new messages
                current_messages = await self.page.query_selector_all(self.SELECTORS["message_container"])
                
                # If we have the expected number of messages, get the assistant's response
                if len(current_messages) >= target_message_count:
                    
                    # The last message should be the assistant's response
                    response_message = current_messages[-1]
                    
                    # Try to find .prose or other content element
                    if response_element is None:
                        response_element = await response_message.query_selector(self.SELECTORS["response_prose"])
                        if not response_element:
                            # No .prose element - try other common selectors in priority order
                            alternatives = [
                                "div.prose",
                                "div.markdown",
                                "div[class*='content']",
                                "div[class*='markdown']",
                                "div[class*='response']",
                                "pre",
                                "p",
                            ]
                            for selector in alternatives:
                                response_element = await response_message.query_selector(selector)
                                if response_element:
                                    test_content = await response_element.inner_text()
                                    if test_content and len(test_content.strip()) > 1:
                                        break
                                    else:
                                        response_element = None
                            
                            if not response_element:
                                # Try finding any div with meaningful content
                                all_divs = await response_message.query_selector_all("div")
                                for div in all_divs:
                                    test_content = await div.inner_text()
                                    if test_content and len(test_content.strip()) > 10:
                                        response_element = div
                                        break
                            
                            if not response_element:
                                # Last resort - use message element itself
                                response_element = response_message
                    
                    if response_element:
                        # Get current content
                        current_content = await response_element.inner_text() or ""
                        
                        # Record TTFT on first non-empty content
                        if first_token_time is None and len(current_content.strip()) > 0:
                            first_token_time = time.time()
                        
                        content = current_content
                        
                        # Check if content has stabilized (streaming complete)
                        if len(content) == last_content_length and len(content) > 0:
                            stable_count += 1
                            if stable_count >= max_stable_checks:
                                # Content hasn't changed for 500ms, response is complete
                                break
                        else:
                            stable_count = 0
                            last_content_length = len(content)
                
                await asyncio.sleep(0.1)  # Poll every 100ms
            
            end_time = time.time()
            
            # Calculate metrics
            total_duration_ms = (end_time - send_time) * 1000
            ttft_ms = (first_token_time - send_time) * 1000 if first_token_time else total_duration_ms
            
            # Estimate tokens (rough approximation: ~4 chars per token)
            tokens_rendered = len(content) // 4 if content else 0
            
            return BrowserChatResult(
                content=content,
                ttft_ms=ttft_ms,
                total_duration_ms=total_duration_ms,
                tokens_rendered=tokens_rendered,
                success=True,
            )
            
        except Exception as e:
            end_time = time.time()
            return BrowserChatResult(
                content="",
                ttft_ms=0,
                total_duration_ms=(end_time - start_time) * 1000,
                tokens_rendered=0,
                success=False,
                error=str(e),
            )
    
    async def take_screenshot(self, path: str) -> None:
        """Take a screenshot of the current page."""
        await self.page.screenshot(path=path)


class BrowserPool:
    """
    Pool of browser clients for concurrent UI benchmark operations.
    
    Manages multiple browser contexts for simulating concurrent users.
    Can use either separate browser contexts (lighter, shared process) or
    separate browser instances (full isolation, heavier).
    """
    
    def __init__(
        self,
        base_url: str,
        headless: bool = True,
        slow_mo: int = 0,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        timeout: float = 30000,
        use_isolated_browsers: bool = False,
    ):
        """
        Initialize the browser pool.
        
        Args:
            base_url: Base URL of the Open WebUI instance
            headless: Run browsers in headless mode
            slow_mo: Slow down operations by ms (for debugging)
            viewport_width: Browser viewport width
            viewport_height: Browser viewport height
            timeout: Default timeout in milliseconds
            use_isolated_browsers: Use separate browser instances (vs contexts)
        """
        self.base_url = base_url
        self.headless = headless
        self.slow_mo = slow_mo
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.timeout = timeout
        self.use_isolated_browsers = use_isolated_browsers
        
        self._playwright: Optional[Playwright] = None
        self._shared_browser: Optional[Browser] = None
        self._clients: List[BrowserClient] = []
        self._user_credentials: List[Dict[str, str]] = []
    
    async def initialize(self) -> None:
        """Initialize the browser pool (start Playwright and shared browser)."""
        self._playwright = await async_playwright().start()
        
        if not self.use_isolated_browsers:
            # Launch a shared browser instance
            self._shared_browser = await self._playwright.chromium.launch(
                headless=self.headless,
                slow_mo=self.slow_mo,
            )
    
    async def create_clients(
        self,
        credentials: List[Dict[str, str]],
        login: bool = True,
        batch_size: int = 10,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[BrowserClient]:
        """
        Create browser clients for each set of credentials.
        
        Creates contexts in batches to avoid overwhelming the server.
        
        Args:
            credentials: List of dicts with 'email' and 'password' keys
            login: Whether to login each client immediately
            batch_size: Number of browsers to create/login concurrently
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            List of initialized (and optionally logged-in) browser clients
        """
        self._user_credentials = credentials
        self._clients = []
        
        async def create_context() -> BrowserClient:
            """Create a browser client with context (no login yet)."""
            if self.use_isolated_browsers:
                client = BrowserClient(
                    base_url=self.base_url,
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    viewport_width=self.viewport_width,
                    viewport_height=self.viewport_height,
                    timeout=self.timeout,
                )
                await client.launch()
            else:
                client = BrowserClient(
                    base_url=self.base_url,
                    headless=self.headless,
                    slow_mo=self.slow_mo,
                    viewport_width=self.viewport_width,
                    viewport_height=self.viewport_height,
                    timeout=self.timeout,
                )
                client._playwright = self._playwright
                client._browser = self._shared_browser
                client._context = await self._shared_browser.new_context(
                    viewport={"width": self.viewport_width, "height": self.viewport_height},
                )
                client._page = await client._context.new_page()
                client._page.set_default_timeout(self.timeout)
            return client
        
        # Step 1: Create all browser contexts concurrently (this is fast)
        context_tasks = [create_context() for _ in credentials]
        self._clients = await asyncio.gather(*context_tasks)
        
        if not login:
            return self._clients
        
        # Step 2: Login in batches to avoid overwhelming the auth endpoint
        total = len(credentials)
        logged_in = 0
        
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_clients = self._clients[batch_start:batch_end]
            batch_creds = credentials[batch_start:batch_end]
            
            # Login this batch concurrently
            async def login_client(client: BrowserClient, cred: Dict[str, str]):
                await client.login(cred["email"], cred["password"])
            
            login_tasks = [
                login_client(client, cred)
                for client, cred in zip(batch_clients, batch_creds)
            ]
            await asyncio.gather(*login_tasks)
            
            logged_in += len(batch_clients)
            if progress_callback:
                progress_callback(logged_in, total)
            
            # Small delay between batches to let the server breathe
            if batch_end < total:
                await asyncio.sleep(0.5)
        
        return self._clients
    
    @property
    def clients(self) -> List[BrowserClient]:
        """Get the list of browser clients."""
        return self._clients
    
    async def close_all(self) -> None:
        """Close all browser clients and clean up resources."""
        for client in self._clients:
            try:
                if self.use_isolated_browsers:
                    await client.close()
                else:
                    # Only close the context, not the shared browser
                    if client._context:
                        await client._context.close()
                        client._context = None
                        client._page = None
            except Exception:
                pass
        
        self._clients = []
        
        if self._shared_browser:
            await self._shared_browser.close()
            self._shared_browser = None
        
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
