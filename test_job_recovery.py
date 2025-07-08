#!/usr/bin/env python3
"""
Test script for job recovery functionality
"""

import sys
import os
sys.path.insert(0, 'src')

from site_management_api import CrawlJobManager
from crawl_progress import CrawlProgressManager, CrawlJobStatus
from pathlib import Path
import tempfile

def test_job_recovery():
    """Test that jobs can be recovered after API server restart"""
    
    print("Testing job recovery functionality...")
    
    # Create a temporary directory for progress files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Override the progress directory for testing
        original_progress_dir = Path("data/progress")
        test_progress_dir = Path(temp_dir) / "progress"
        test_progress_dir.mkdir(exist_ok=True)
        
        # Create first job manager instance
        job_manager1 = CrawlJobManager()
        
        # Create a test job
        site_configs = {
            'test_site': {
                'name': 'Test Site',
                'limits': {'max_articles': 100, 'max_depth': 3}
            }
        }
        
        job_id = job_manager1.create_job(['test_site'], site_configs)
        print(f"Created job: {job_id}")
        
        # Verify job exists
        job = job_manager1.get_job(job_id)
        assert job is not None, "Job should exist after creation"
        print(f"Job status: {job['status']}")
        
        # Simulate job running
        job_manager1.update_job(job_id, {'status': 'running'})
        progress = job_manager1.get_job_progress(job_id)
        if progress:
            progress.status = CrawlJobStatus.RUNNING
            job_manager1.progress_manager.save_progress(job_id)
        
        # Simulate pausing the job
        success = job_manager1.progress_manager.pause_job(job_id)
        assert success, "Should be able to pause running job"
        print("Job paused successfully")
        
        # Create second job manager instance (simulating API server restart)
        job_manager2 = CrawlJobManager()
        
        # Test that job is loaded from disk
        job_loaded = job_manager2.get_job(job_id)
        if job_loaded:
            print(f"Job loaded successfully: {job_loaded['status']}")
        else:
            print("Job not loaded initially, testing ensure_job_loaded...")
            loaded = job_manager2.ensure_job_loaded(job_id)
            assert loaded, "Should be able to load job from disk"
            job_loaded = job_manager2.get_job(job_id)
            assert job_loaded is not None, "Job should exist after loading"
            print(f"Job loaded with ensure_job_loaded: {job_loaded['status']}")
        
        # Test that we can get job progress
        progress = job_manager2.get_job_progress(job_id)
        assert progress is not None, "Should be able to get job progress"
        print(f"Job progress status: {progress.status.value}")
        
        # Test resuming the job
        success = job_manager2.progress_manager.resume_job(job_id)
        assert success, "Should be able to resume paused job"
        print("Job resumed successfully")
        
        # Test stopping the job
        success = job_manager2.progress_manager.stop_job(job_id)
        assert success, "Should be able to stop job"
        print("Job stopped successfully")
        
        print("✅ All job recovery tests passed!")
        return True

if __name__ == "__main__":
    try:
        test_job_recovery()
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)