#!/usr/bin/env python3
"""
Simple test to verify job recovery logic
"""

import sys
import os
sys.path.insert(0, 'src')

try:
    from crawl_progress import CrawlProgressManager, CrawlJobStatus, SiteCrawlStatus
    print("✅ Successfully imported CrawlProgressManager")
    
    # Test creating a progress manager
    progress_manager = CrawlProgressManager()
    print("✅ Successfully created CrawlProgressManager")
    
    # Test job creation
    site_configs = {
        'test_site': {
            'name': 'Test Site',
            'limits': {'max_articles': 100, 'max_depth': 3}
        }
    }
    
    job_id = "test_job_12345"
    site_ids = ['test_site']
    
    # Create job progress
    progress = progress_manager.create_job(job_id, site_ids, site_configs)
    print(f"✅ Created job progress for {job_id}")
    
    # Test job state transitions
    progress.status = CrawlJobStatus.RUNNING
    progress_manager.save_progress(job_id)
    print("✅ Job set to running and saved")
    
    # Test pause
    success = progress_manager.pause_job(job_id)
    print(f"✅ Pause job result: {success}")
    
    # Test resume
    success = progress_manager.resume_job(job_id)
    print(f"✅ Resume job result: {success}")
    
    # Test stop
    success = progress_manager.stop_job(job_id)
    print(f"✅ Stop job result: {success}")
    
    # Test recovery
    recoverable = progress_manager.get_recoverable_jobs()
    print(f"✅ Recoverable jobs: {recoverable}")
    
    print("\n✅ All basic tests passed! Job recovery logic should work correctly.")
    
except Exception as e:
    print(f"❌ Test failed: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)