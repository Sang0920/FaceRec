import os
import json
import shutil
import requests
import logging
import sys
from datetime import datetime
from dotenv import load_dotenv

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # logging.FileHandler('api.log'),
        logging.FileHandler(os.path.join('logs', 'api.log')), # Change to logs/api.log
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DracoAPIClient:
    def __init__(self):
        load_dotenv()
        self.api_url = os.getenv('API_URL')
        self.base_url = os.getenv('BASE_URL')
        self.api_key = os.getenv('API_KEY')
        self.api_secret = os.getenv('API_SECRET')
        
        if not all([self.api_url, self.base_url, self.api_key, self.api_secret]):
            logger.error("Missing required environment variables")
            raise ValueError("Missing required environment variables")
            
        self.headers = {
            "Authorization": f"token {self.api_key}:{self.api_secret}",
            "Content-Type": "application/json"
        }
        logger.info("DracoAPIClient initialized successfully")

    def create_checkin(self, email, timestamp, log_type="IN", image_base64=None, pdf_base64=None):
        """Create employee check-in record with logging"""
        url = f"{self.api_url}/hrms.hr.doctype.employee_checkin.employee_checkin.create_employee_checkin"
        data = {
            "email": email,
            "timestamp": timestamp,
            "log_type": log_type,
            "image_base64": image_base64,
            "pdf_base64": pdf_base64
        }
        
        try:
            logger.info(f"Creating check-in for {email} at {timestamp}")
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            result = response.json() # {'message': {'status': 'success', 'message': 'Check-in recorded', 'checkin_id': 'EMP-CKIN-02-2025-000011', 'pdf_url': None}}
            
            # if result.get('message', {}).get('name'):
            if result.get('message', {}).get('status') == 'success':
                logger.info(f"✅ Check-in successful - {email} - Response: {result}")
                return True, result
            else:
                logger.error(f"❌ Check-in failed - {email} - Response: {result}")
                return False, result
                
        except Exception as e:
            logger.error(f"❌ Error creating check-in - {email}: {str(e)} - Data: {data} - Response: {response.text}")
            return False, {"error": str(e)}

    def get_employee_photos(self, branch: str = None):
        """Get employee photos with logging"""
        url = f"{self.api_url}/draerp.setup.doctype.employee.employee.get_employee_photos"
        params = {"branch": branch} if branch else {}
        
        try:
            logger.info(f"Fetching employee photos{' for ' + branch if branch else ''}")
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            photos = data.get("message", {})
            
            if photos:
                logger.info(f"✅ Successfully retrieved {len(photos)} employee photos")
                return photos
            else:
                logger.warning("No photo data found in response")
                return {}
                
        except Exception as e:
            logger.error(f"❌ Error fetching employee photos: {str(e)}")
            return {}

    def sync_employee_photos(self, base_dir="faces"):
        """Synchronize employee photos with logging"""
        logger.info("Starting employee photo sync")
        os.makedirs(base_dir, exist_ok=True)
        metadata_file = "sync_metadata.json"
        
        try:
            # Load metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded existing metadata for {len(metadata)} employees")
            else:
                metadata = {}
                logger.info("No existing metadata found, starting fresh")

            # Get current photos
            photo_map = self.get_employee_photos()
            
            # Process removals
            current_emails = set(photo_map.keys())
            existing_emails = set(metadata.keys())
            removed = existing_emails - current_emails
            
            for email in removed:
                email_dir = os.path.join(base_dir, email)
                if os.path.exists(email_dir):
                    shutil.rmtree(email_dir)
                    logger.info(f"Removed folder for ex-employee: {email}")
                metadata.pop(email, None)

            # Process updates
            for email, photo_path in photo_map.items():
                try:
                    email_dir = os.path.join(base_dir, email)
                    os.makedirs(email_dir, exist_ok=True)
                    
                    photo_url = f"{self.base_url}{photo_path}"
                    last_modified = metadata.get(email, {}).get('last_modified', '')
                    
                    response = requests.head(photo_url)
                    current_modified = response.headers.get('Last-Modified', 
                                                         datetime.now().isoformat())
                    
                    if last_modified != current_modified:
                        image_response = requests.get(photo_url)
                        if image_response.status_code == 200:
                            photo_file = os.path.join(email_dir, "profile.jpg")
                            with open(photo_file, 'wb') as f:
                                f.write(image_response.content)
                            metadata[email] = {
                                'last_modified': current_modified,
                                'last_synced': datetime.now().isoformat()
                            }
                            logger.info(f"✅ Updated photo for: {email}")
                        else:
                            logger.error(f"❌ Failed to download photo for: {email}")
                            
                except Exception as e:
                    logger.error(f"❌ Error processing {email}: {str(e)}")

            # Save metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.info("✅ Photo sync completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during photo sync: {str(e)}")

if __name__ == "__main__":
    client = DracoAPIClient()
    # Example usage:
    # client.sync_employee_photos()
    client.create_checkin("sangdt@draco.biz", "2025-02-03 08:10:11")