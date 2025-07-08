#!/usr/bin/env python3
"""
Authentication Manager for Multi-Site RAG System

This module handles SSO authentication for sites that require login,
managing browser profiles, session validation, and authentication workflows.
"""

import os
import json
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from crawl4ai import AsyncWebCrawler, BrowserConfig
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from site_config import SiteConfig, AuthConfig

logger = logging.getLogger(__name__)


@dataclass
class AuthStatus:
    """Authentication status for a site"""
    is_authenticated: bool = False
    last_check: Optional[datetime] = None
    session_expires: Optional[datetime] = None
    error_message: Optional[str] = None
    profile_path: Optional[str] = None


class AuthenticationManager:
    """Manages SSO authentication for protected sites"""
    
    def __init__(self, auth_data_dir: str = "data/auth"):
        """Initialize authentication manager
        
        Args:
            auth_data_dir: Directory to store authentication data and browser profiles
        """
        self.auth_data_dir = Path(auth_data_dir)
        self.auth_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Directory for browser profiles
        self.profiles_dir = self.auth_data_dir / "profiles"
        self.profiles_dir.mkdir(exist_ok=True)
        
        # File to store authentication status
        self.status_file = self.auth_data_dir / "auth_status.json"
        
        # Load existing authentication status
        self.auth_status: Dict[str, AuthStatus] = {}
        self._load_auth_status()
    
    def _load_auth_status(self):
        """Load authentication status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for site_id, status_data in data.items():
                    # Convert datetime strings back to datetime objects
                    if status_data.get('last_check'):
                        status_data['last_check'] = datetime.fromisoformat(status_data['last_check'])
                    if status_data.get('session_expires'):
                        status_data['session_expires'] = datetime.fromisoformat(status_data['session_expires'])
                    
                    self.auth_status[site_id] = AuthStatus(**status_data)
                
                logger.info(f"Loaded authentication status for {len(self.auth_status)} sites")
                
            except Exception as e:
                logger.error(f"Error loading authentication status: {str(e)}")
                self.auth_status = {}
    
    def _save_auth_status(self):
        """Save authentication status to file"""
        try:
            data = {}
            for site_id, status in self.auth_status.items():
                status_dict = {
                    'is_authenticated': status.is_authenticated,
                    'last_check': status.last_check.isoformat() if status.last_check else None,
                    'session_expires': status.session_expires.isoformat() if status.session_expires else None,
                    'error_message': status.error_message,
                    'profile_path': status.profile_path
                }
                data[site_id] = status_dict
            
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving authentication status: {str(e)}")
    
    def get_profile_path(self, site_config: SiteConfig) -> str:
        """Get the browser profile path for a site
        
        Args:
            site_config: Site configuration
            
        Returns:
            Path to the browser profile directory
        """
        if site_config.auth_config.user_data_dir:
            return site_config.auth_config.user_data_dir
        
        # Generate default profile path
        safe_name = "".join(c for c in site_config.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        profile_path = self.profiles_dir / f"{safe_name}_{site_config.id[:8]}"
        return str(profile_path)
    
    async def check_authentication_status(self, site_config: SiteConfig) -> AuthStatus:
        """Check if the site is currently authenticated
        
        Args:
            site_config: Site configuration
            
        Returns:
            Current authentication status
        """
        if not site_config.auth_config.requires_sso:
            return AuthStatus(is_authenticated=True, last_check=datetime.now())
        
        profile_path = self.get_profile_path(site_config)
        
        # Check if we have cached status that's still valid
        status = self.auth_status.get(site_config.id)
        if status and status.session_expires and datetime.now() < status.session_expires:
            logger.info(f"Using cached authentication status for {site_config.name}")
            return status
        
        # Test authentication by accessing the auth test URL or a start URL
        test_url = site_config.auth_config.auth_test_url or site_config.start_urls[0] if site_config.start_urls else site_config.base_url
        
        try:
            browser_config = BrowserConfig(
                headless=True,
                user_data_dir=profile_path,
                browser_type="chromium"
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(
                    url=test_url,
                    wait_for="body",
                    timeout=30000
                )
                
                # Check for authentication indicators
                is_authenticated = self._check_auth_indicators(result.html, site_config)
                
                # Calculate session expiry
                session_expires = datetime.now() + timedelta(hours=site_config.auth_config.session_timeout_hours)
                
                status = AuthStatus(
                    is_authenticated=is_authenticated,
                    last_check=datetime.now(),
                    session_expires=session_expires,
                    profile_path=profile_path,
                    error_message=None if is_authenticated else "Authentication check failed"
                )
                
                self.auth_status[site_config.id] = status
                self._save_auth_status()
                
                logger.info(f"Authentication check for {site_config.name}: {'✓' if is_authenticated else '✗'}")
                return status
                
        except Exception as e:
            logger.error(f"Error checking authentication for {site_config.name}: {str(e)}")
            status = AuthStatus(
                is_authenticated=False,
                last_check=datetime.now(),
                error_message=str(e),
                profile_path=profile_path
            )
            self.auth_status[site_config.id] = status
            self._save_auth_status()
            return status
    
    def _check_auth_indicators(self, html: str, site_config: SiteConfig) -> bool:
        """Check HTML for authentication indicators
        
        Args:
            html: HTML content to check
            site_config: Site configuration
            
        Returns:
            True if authenticated, False otherwise
        """
        # Common authentication failure indicators
        auth_failure_indicators = [
            "sign in", "log in", "login", "authenticate",
            "unauthorized", "access denied", "please login",
            "session expired", "not authorized"
        ]
        
        html_lower = html.lower()
        
        # If we find login/auth failure indicators, probably not authenticated
        for indicator in auth_failure_indicators:
            if indicator in html_lower:
                return False
        
        # If we can access content that should be protected, probably authenticated
        if any(selector in html_lower for selector in ["dashboard", "profile", "settings", "account"]):
            return True
        
        # If we have specific content selectors and they're present, probably authenticated
        if site_config.selectors.content:
            # Simple check for content selector presence (basic heuristic)
            content_selectors = site_config.selectors.content.split(',')
            for selector in content_selectors:
                selector = selector.strip().replace('.', '').replace('#', '')
                if selector.lower() in html_lower:
                    return True
        
        # Default to not authenticated if we can't determine
        return False
    
    async def setup_authentication(self, site_config: SiteConfig, headless: bool = False) -> bool:
        """Setup authentication for a site by opening a browser for manual login
        
        Args:
            site_config: Site configuration
            headless: Whether to run browser in headless mode
            
        Returns:
            True if setup was successful, False otherwise
        """
        if not site_config.auth_config.requires_sso:
            logger.info(f"Site {site_config.name} doesn't require SSO")
            return True
        
        profile_path = self.get_profile_path(site_config)
        login_url = site_config.auth_config.login_url or site_config.base_url
        
        logger.info(f"Setting up authentication for {site_config.name}")
        logger.info(f"Profile path: {profile_path}")
        logger.info(f"Login URL: {login_url}")
        
        try:
            browser_config = BrowserConfig(
                headless=headless,
                user_data_dir=profile_path,
                browser_type="chromium"
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                # Navigate to login page
                result = await crawler.arun(
                    url=login_url,
                    wait_for="body",
                    timeout=60000
                )
                
                if not headless:
                    print(f"\n🔐 Browser opened for manual authentication")
                    print(f"📍 Navigate to: {login_url}")
                    print(f"🔑 Complete the SSO login process")
                    print(f"✅ Close the browser when login is complete")
                    print(f"💾 Your session will be saved to: {profile_path}")
                    
                    # Keep the browser open for manual interaction
                    input("\nPress Enter after you've completed the login process...")
                
                # Verify authentication was successful
                auth_status = await self.check_authentication_status(site_config)
                
                if auth_status.is_authenticated:
                    logger.info(f"✅ Authentication setup successful for {site_config.name}")
                    return True
                else:
                    logger.warning(f"❌ Authentication setup failed for {site_config.name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error setting up authentication for {site_config.name}: {str(e)}")
            return False
    
    def get_auth_status(self, site_id: str) -> Optional[AuthStatus]:
        """Get authentication status for a site
        
        Args:
            site_id: Site ID
            
        Returns:
            Authentication status or None if not found
        """
        return self.auth_status.get(site_id)
    
    def clear_auth_status(self, site_id: str) -> bool:
        """Clear authentication status for a site
        
        Args:
            site_id: Site ID
            
        Returns:
            True if cleared successfully
        """
        if site_id in self.auth_status:
            del self.auth_status[site_id]
            self._save_auth_status()
            logger.info(f"Cleared authentication status for site {site_id}")
            return True
        return False
    
    def get_browser_config(self, site_config: SiteConfig) -> Optional[BrowserConfig]:
        """Get browser configuration for authenticated crawling
        
        Args:
            site_config: Site configuration
            
        Returns:
            Browser configuration with authentication profile
        """
        if not site_config.auth_config.requires_sso:
            return None
        
        profile_path = self.get_profile_path(site_config)
        
        return BrowserConfig(
            headless=True,
            user_data_dir=profile_path,
            browser_type="chromium",
            wait_for="body"
        )
    
    def list_auth_sites(self) -> Dict[str, AuthStatus]:
        """List all sites with authentication status
        
        Returns:
            Dictionary of site IDs to authentication status
        """
        return self.auth_status.copy()