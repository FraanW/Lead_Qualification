from typing import Union
from crewai.tools import BaseTool
import requests


class WebCrawlTool(BaseTool):
    name: str = "web_crawl"
    description: str = "Crawl a website to extract text content, targeting header/footer, 'Contact Us' pages, and main business info."

    def _run(self, url: Union[str, dict] = None, **kwargs) -> str:
        try:
            import re
            from urllib.parse import urljoin
            
            # Handle cases where the agent passes a dictionary or unexpected keyword arguments
            if isinstance(url, dict):
                url = url.get('url') or url.get('target_url') or url.get('website') or str(url)
            elif url is None and kwargs:
                url = kwargs.get('url') or kwargs.get('target_url') or kwargs.get('website') or next(iter(kwargs.values()))
            
            if not url or not isinstance(url, str):
                return "Error: No valid URL provided to crawl."

            # Strip whitespace and reject non-HTTP URLs
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                return f"Error: Invalid URL '{url}'. URL must start with http:// or https://."

            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, timeout=10, headers=headers)
            response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 1. Discover "Contact Us" or "About" links
            contact_url = None
            for link in soup.find_all('a', href=True):
                text = link.get_text().lower()
                href = link['href'].lower()
                if re.search(r'contact|about-us|find-us|locations', text) or re.search(r'contact|about|location', href):
                    contact_url = urljoin(url, link['href'])
                    break
            
            # Clean up noise from homepage
            for tag in soup(["script", "style", "svg", "path", "iframe"]):
                tag.decompose()
            
            header = soup.find(['header', 'div'], id=re.compile(r'header', re.I), class_=re.compile(r'header', re.I))
            if not header: header = soup.find('header')
            
            footer = soup.find(['footer', 'div'], id=re.compile(r'footer', re.I), class_=re.compile(r'footer', re.I))
            if not footer: footer = soup.find('footer')
            
            header_text = header.get_text(separator=' ').strip() if header else ""
            footer_text = footer.get_text(separator=' ').strip() if footer else ""
            
            if not header_text:
                header_alt = soup.find(['div', 'nav'], class_=re.compile(r'header|top|nav', re.I))
                if header_alt: header_text = header_alt.get_text(separator=' ').strip()
                
            if not footer_text:
                footer_alt = soup.find(['div', 'section'], class_=re.compile(r'footer|bottom', re.I))
                if footer_alt: footer_text = footer_alt.get_text(separator=' ').strip()

            full_text = soup.get_text(separator=' ')
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            main_text = ' '.join(lines)[:2500] # Increased for address context
            
            result = f"DOMAIN: {url}\n\n"
            result += f"--- WEBSITE HEADER (Potential Contacts/Links) ---\n{header_text[:1000] if header_text else 'No header.'}\n\n"
            result += f"--- WEBSITE FOOTER (Potential Addresses/Contacts) ---\n{footer_text[:1000] if footer_text else 'No footer.'}\n\n"
            result += f"--- MAIN PAGE CONTENT (Business Focus & Addresses) ---\n{main_text}\n\n"
            
            # 2. Crawl Contact Page if found
            if contact_url and contact_url != url:
                try:
                    c_res = requests.get(contact_url, timeout=8, headers=headers)
                    if c_res.status_code == 200:
                        c_soup = BeautifulSoup(c_res.text, 'html.parser')
                        for tag in c_soup(["script", "style", "svg", "path", "iframe"]):
                            tag.decompose()
                        c_text = c_soup.get_text(separator=' ')
                        c_lines = [line.strip() for line in c_text.splitlines() if line.strip()]
                        result += f"--- DEDICATED CONTACT PAGE ({contact_url}) ---\n"
                        result += ' '.join(c_lines)[:2000]
                except:
                    result += f"\n(Note: Failed to crawl found contact page: {contact_url})"
            
            return result
        except Exception as e:
            return f"Crawl Error: {str(e)}"
