from crewai.tools import BaseTool
import requests


# Third-party resellers, marketplaces, aggregators, and directories
# These sell/list many brands but are never the official brand website
THIRD_PARTY_DOMAINS = [
    # Social media
    'linkedin', 'facebook', 'instagram', 'twitter', 'tiktok',
    # Job sites
    'indeed', 'glassdoor', 'bayt.com',
    # Reference sites
    'wikipedia', 'youtube', 'pinterest',
    # Resellers & marketplaces
    'theluxurycloset', 'noon.com', 'namshi', 'amazon', 'ebay',
    'ounass', 'farfetch', 'net-a-porter', 'mytheresa', 'ssense',
    'yoox', 'matchesfashion', 'shopbop', 'nordstrom',
    'bloomingdales', 'vinted', 'depop', 'asos',
    # UAE aggregators & directories
    'yellowpages', 'bayut', 'propertyfinder', 'dubizzle',
    'zomato', 'talabat', 'carrefouruae', '2gis', 'refdubai', 'dubaipicks',
    # General directories
    'crunchbase', 'zoominfo', 'dnb.com', 'bloomberg',
]


class SearXNGSearchTool(BaseTool):
    name: str = "searxng_search"
    description: str = "Search the web using a local metasearch engine. Returns structured company data."
    
    # Points to your WSL instance running on port 8888
    searx_host: str = "http://localhost:8888" 
    brand_name_filter: str = "" # Added for filtering

    def _score_results(self, raw_results: list, brand_variants: list, brand_words_list: list) -> list:
        """Score search results to find the most likely official brand website.
        
        Args:
            raw_results: Search results from the search engine.
            brand_variants: List of normalized brand name variants to try matching
                           (e.g. ['maxandco', 'maxco'] for 'MAX&Co.').
            brand_words_list: List of lists of brand word variants for word-level matching.
        """
        scored_sites = []
        
        for res in raw_results[:15]:
            url = res.get('url', '').lower()
            domain = url.split('//')[-1].split('/')[0]
            
            if not any(excluded in domain for excluded in THIRD_PARTY_DOMAINS):
                domain_clean = domain.replace('.', '').replace('-', '')
                
                # Try all brand name variants and pick the best score
                best_variant_score = 0
                for variant in brand_variants:
                    score = 0
                    
                    # Exact full brand name in domain (high score)
                    if variant and variant in domain_clean:
                        score += 50
                    
                    if score > best_variant_score:
                        best_variant_score = score
                
                score = best_variant_score
                
                # Individual words match in domain (moderate boost)
                # Use the best word list match
                best_word_score = 0
                for words in brand_words_list:
                    word_score = sum(20 for w in words if w in domain)
                    if word_score > best_word_score:
                        best_word_score = word_score
                score += best_word_score
                
                # UAE focus — only check domain, not full URL path
                if '.ae' in domain or 'uae' in domain or 'dubai' in domain:
                    score += 40
                # Shorter URLs preferred for official sites
                score -= (len(url.split('/')) - 3) * 10
                
                # Penalty: brand appears in URL path but NOT in domain
                url_path = url.split('//', 1)[-1].split('/', 1)[-1] if '/' in url.split('//', 1)[-1] else ''
                path_normalized = url_path.replace('-', '').replace('_', '').replace('/', '')
                brand_in_domain = any(v and v in domain_clean for v in brand_variants)
                brand_in_path = any(v and v in path_normalized for v in brand_variants)
                if brand_in_path and not brand_in_domain:
                    score -= 40
                
                # Penalty for short brand names: if no variant fully matches the
                # domain, cap the score (prevents 'maxfashion' matching 'maxco')
                shortest_variant = min((len(v) for v in brand_variants if v), default=0)
                if shortest_variant and shortest_variant <= 6 and not brand_in_domain and score < 50:
                    score = min(score, 5)
                
                if score > 0:
                    scored_sites.append((score, res.get('url')))
        
        scored_sites.sort(key=lambda x: x[0], reverse=True)
        return scored_sites

    def _do_search(self, query: str) -> list:
        """Execute a search and return raw results."""
        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,duckduckgo,qwant",
            "categories": "general",
            "safesearch": 0,
            "language": "en-US"
        }
        response = requests.get(f"{self.searx_host}/search", params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])

    def _run(self, query: str = None, **kwargs) -> str:
        try:
            # Handle cases where the agent passes a dictionary or kwargs
            if isinstance(query, dict):
                 query = query.get('query') or query.get('q') or str(query)
            elif query is None and kwargs:
                 query = kwargs.get('query') or kwargs.get('q') or next(iter(kwargs.values()), '')
            
            # Guard against empty/whitespace-only queries
            if not query or not isinstance(query, str) or not query.strip():
                return f"Search Error: Empty query provided. Please provide a search query string."

            brand_kw = self.brand_name_filter.lower() if self.brand_name_filter else ""
            
            # Generate TWO normalizations to handle '&' in brand names:
            # "MAX&Co." → variant1: "maxandco" (& → and), variant2: "maxco" (& stripped)
            # This ensures we match domains like "maxandco.com" AND "hm.com" (for "H&M")
            brand_with_and = brand_kw.replace('&', 'and').replace('+', 'plus')
            brand_variant1 = ''.join(c for c in brand_with_and if c.isalnum())  # & → and
            brand_variant2 = ''.join(c for c in brand_kw if c.isalnum())        # & stripped
            
            # Deduplicate variants (they may be the same if no & in name)
            brand_variants = list(dict.fromkeys([brand_variant1, brand_variant2]))
            
            # Build word lists for each variant
            brand_words_list = []
            for variant in brand_variants:
                words = [w for w in variant.split() if len(w) > 2]
                if not words and variant:
                    words = [variant]
                brand_words_list.append(words)

            # === PRIMARY SEARCH ===
            raw_results = self._do_search(query)
            
            # Filter to business-relevant snippets for display
            filtered_results = [
                r for r in raw_results 
                if any(kw in (r.get('title', '') + r.get('content', '')).lower() 
                       for kw in ['uae', 'gcc', 'business', 'company', 'contact', 'phone', 'email', 'maps', brand_kw])
            ]
            results = filtered_results[:2] if filtered_results else raw_results[:2]
            
            if not results and not raw_results:
                return "No results found for this business."

            # Score primary results
            scored_sites = self._score_results(raw_results, brand_variants, brand_words_list)
            best_score = scored_sites[0][0] if scored_sites else 0

            # === SECONDARY SEARCH (exact brand name) ===
            # If primary search didn't find a confident official site, try a
            # more targeted search with the exact brand name in quotes
            if best_score < 30 and self.brand_name_filter:
                try:
                    secondary_query = f'"{self.brand_name_filter}" official website'
                    secondary_results = self._do_search(secondary_query)
                    secondary_scored = self._score_results(secondary_results, brand_variants, brand_words_list)
                    
                    if secondary_scored:
                        # Merge with primary results, keeping best scores
                        seen_urls = {url for _, url in scored_sites}
                        for score, url in secondary_scored:
                            if url not in seen_urls:
                                scored_sites.append((score, url))
                        scored_sites.sort(key=lambda x: x[0], reverse=True)
                        best_score = scored_sites[0][0] if scored_sites else 0
                except Exception:
                    pass  # Secondary search is best-effort

            # Pick the highest scored site
            official_website = None
            if scored_sites:
                official_website = scored_sites[0][1]
            
            # Build output
            formatted = []
            if official_website and best_score >= 30:
                formatted.append(f"Official Website (VERIFIED): {official_website}\n")
            elif official_website:
                formatted.append(
                    f"Official Website (LOW CONFIDENCE): {official_website}\n"
                    f"WARNING: This URL may not be the official website for '{self.brand_name_filter}'. "
                    f"Verify the domain matches the exact brand name before using it.\n"
                )
            else:
                formatted.append(
                    f"WARNING: No official website found for '{self.brand_name_filter}'. "
                    f"Do NOT use URLs from search snippets below as the official website — "
                    f"they may belong to different brands or third-party sites.\n"
                )
            
            for res in results:
                formatted.append(
                    f"Title: {res.get('title')}\n"
                    f"Description: {res.get('content')}\n"
                    f"URL: {res.get('url')}\n"
                )
            
            return "\n---\n".join(formatted)
        except Exception as e:
            return f"Local Search Error: {str(e)}"
