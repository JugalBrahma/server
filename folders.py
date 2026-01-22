"""
Folders Module - Cleaned Up
Router removed since we have test routers in test folder
"""

# This file is kept for backward compatibility
# All folder functionality moved to test folder routers
# Use test/simple_upload_server.py for file upload testing
# Use test/test_router.py for API testing
# Original router backed up in folders_backup.py

def get_info():
    """Get info about this module"""
    return {
        "message": "Folders module has been cleaned up",
        "status": "router_removed",
        "alternatives": {
            "file_upload": "test/simple_upload_server.py",
            "api_testing": "test/test_router.py", 
            "rag_testing": "test/simple_test_server.py"
        }
    }

if __name__ == "__main__":
    info = get_info()

