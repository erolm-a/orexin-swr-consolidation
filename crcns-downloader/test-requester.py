import os
import ssl
import sys
import http.cookiejar
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

# --- Updated to use httpbun.com as the active alternative ---
COOKIE_FILE = "test_cookies.txt"
source_cookie_name = "data_source"
site_name = "httpbun.com"  # Switched from httpbin.org


class Requester:

    def __init__(self, new_data_source, output_dir):
        self.output_dir = Path(output_dir) 
        self.cookie_jar = http.cookiejar.MozillaCookieJar(COOKIE_FILE)
        if os.path.exists(COOKIE_FILE):
            try:
                self.cookie_jar.load(ignore_discard=True, ignore_expires=True)
            except Exception as e:
                pass


        # Try to use certifi bundle if available, otherwise fallback to defaults
        try:
            import certifi
            print("using certifi")
            ssl_context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            # Fallback for systems that don't have certifi installed
            print("certifi NOT FOUND")
            ssl_context = ssl.create_default_context()

        # Create the handler using our robust context
        ssl_handler = urllib.request.HTTPSHandler(context=ssl_context)

        """
        # --- CROSS-PLATFORM SSL HANDLER SETUP ---
        ssl_context = ssl.create_default_context()
        if sys.platform == "win32":
            try:
                ssl_context.load_default_certs(purpose=ssl.Purpose.SERVER_AUTH)
                if hasattr(ssl, 'enum_certificates'):
                    for cert, encoding, trust in ssl.enum_certificates("ROOT"):
                        if trust is True or (isinstance(trust, set) and 'serverAuth' in trust):
                            if encoding == 'der':
                                try:
                                    ssl_context.load_verify_locations(cadata=cert)
                                except Exception:
                                    pass 
            except Exception:
                pass
        ssl_handler = urllib.request.HTTPSHandler(context=ssl_context)
        """

        self.save_data_source(new_data_source)
        self.cookie_handler = urllib.request.HTTPCookieProcessor(self.cookie_jar)
        self.redirect_handler = urllib.request.HTTPRedirectHandler()
        
        self.opener = urllib.request.build_opener(
            self.cookie_handler, 
            self.redirect_handler, 
            ssl_handler
        )
        
        self.opener.addheaders = [
            ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                ' AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
        ]

    def fetch_small(self, url, form_vars=None):
        html = None
        status_code = 0
        try:
            post_data = urllib.parse.urlencode(form_vars).encode('utf-8') if form_vars else None
            with self.opener.open(url, data=post_data) as response:
                html = response.read().decode('utf-8')
                status_code = response.getcode()
            self.save_cookies()
        except urllib.error.URLError as e:
            print(f"HTTP Request failed: {e.reason}")
        return (html, status_code)

    def save_data_source(self, new_data_source):
        previous_data_source = self.get_cookie_value(source_cookie_name, site_name)
        if(new_data_source is None):
            if(previous_data_source is not None):
                print(f"Datasource is '{previous_data_source}'.")
            return None
        if(previous_data_source != new_data_source):
            if (previous_data_source is None):
                print(f"Data source set to '{new_data_source}'.")
            else:
                print(f"Data source changed from '{previous_data_source}' to '{new_data_source}'.")
            if(new_data_source == ""):
                self.cookie_jar.clear(domain=site_name, path='/', name=source_cookie_name)
            else:
                self.add_cookie(source_cookie_name, new_data_source, site_name)

    def save_cookies(self):
        self.cookie_jar.save(ignore_discard=True, ignore_expires=True)

    def get_cookie_value(self, cookie_name, domain):
        for cookie in self.cookie_jar:
            if cookie.name == cookie_name and cookie.domain == domain:
                return cookie.value
        return None

    def add_cookie(self, name, value, domain, path='/', secure=False):
        cookie = http.cookiejar.Cookie(
            version=0, name=name, value=value, port=None, port_specified=False,
            domain=domain, domain_specified=True, domain_initial_dot=domain.startswith('.'),
            path=path, path_specified=True, secure=secure, expires=None, discard=True,
            comment=None, comment_url=None, rest={}
        )
        self.cookie_jar.set_cookie(cookie)

    def download_large_file(self, url, output_filename):
        output_file_path = self.output_dir / output_filename
        directory = os.path.dirname(output_file_path)
    
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True) 
            print(f"Created missing directory structure: {directory}")

        downloaded_bytes = 0
        file_mode = 'wb'
        
        if os.path.exists(output_file_path):
            downloaded_bytes = os.path.getsize(output_file_path)
            if downloaded_bytes > 0:
                file_mode = 'ab'
                print(f"Resuming download from byte offset: {downloaded_bytes}")

        req = urllib.request.Request(url)
        if downloaded_bytes > 0:
            req.add_header('Range', f'bytes={downloaded_bytes}-')

        try:
            with self.opener.open(req) as response:
                if response.status == 200 and downloaded_bytes > 0:
                    print("Server ignored Range header. Restarting full download.")
                    file_mode = 'wb'
                    downloaded_bytes = 0
                
                content_length = response.getheader('Content-Length')
                total_size = int(content_length) + downloaded_bytes if content_length else None
                
                chunk_size = 1024 * 1024 
                with open(output_file_path, file_mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        
                        if total_size:
                            percent = (downloaded_bytes / total_size) * 100
                            sys.stdout.write(f"\rProgress: {percent:.2f}%")
                            sys.stdout.flush()

                self.save_cookies() 
                print(f"\n{output_file_path} downloaded successfully.")
                return True
                
        except urllib.error.URLError as e:
            print(f"\nLarge download encountered network error: {e.reason}")
            return False


if __name__ == "__main__":
    print("--- STEP 1: Initializing Requester (Testing Data Source Cookie Assignment) ---")
    output_directory = Path("./test_downloads")
    requester = Requester(new_data_source="HPSS", output_dir=output_directory)

    print("\n--- STEP 2: Testing HTTPS Connection & SSL Validation (`fetch_small`) ---")
    test_url = f"https://{site_name}/get"
    print(f"Attempting to fetch {test_url}...")
    html_response, status = requester.fetch_small(test_url)
    
    if status == 200 and html_response:
        print("Success! SSL handshake passed and content downloaded.")
        print(f"Response Preview (First 150 chars):\n{html_response[:150]}...")
    else:
        print(f"Failure. Received HTTP status: {status}")

    print("\n--- STEP 3: Testing Cookie Persistence & Server Reception ---")
    cookie_test_url = f"https://{site_name}/cookies/set?session_token=abc123xyz"
    print(f"Telling remote server to drop a cookie via: {cookie_test_url}")
    requester.fetch_small(cookie_test_url)
    
    check_cookie_url = f"https://{site_name}/cookies"
    cookie_check_response, _ = requester.fetch_small(check_cookie_url)
    
    # Safe check in case network fails entirely
    if cookie_check_response:
        print(f"Server reports seeing these cookies: {cookie_check_response.strip()}")
    else:
        print("Failed to receive cookie response verification due to previous network issue.")
    
    if os.path.exists(COOKIE_FILE):
        print(f"Confirmed local cookie registry file generated at: '{COOKIE_FILE}'")

    print("\n--- STEP 4: Testing Streaming Binary Files (`download_large_file`) ---")
    # Using Cloudflare's speed testing tool for a 5MB network binary stream
    large_file_url = "https://cloudflare.com" 
    target_filename = "large_test_blob.bin"
    
    print(f"Streaming a 5MB binary block to '{output_directory / target_filename}'...")
    success = requester.download_large_file(large_file_url, target_filename)
    
    if success:
        downloaded_path = output_directory / target_filename
        print(f"Verified target exists on disk: {downloaded_path.exists()}")
        print(f"Final file size: {os.path.getsize(downloaded_path)} bytes")
