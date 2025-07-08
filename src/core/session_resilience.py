#!/usr/bin/env python3
"""
Session-level resilience for memory-intensive EEG analysis.

This module provides robust session processing with automatic retry logic
for handling memory allocation failures gracefully.
"""

import gc
import os
import time
import traceback
from typing import Dict, List, Optional, Tuple, Any
from memory_monitor import force_garbage_collection, monitor_memory_usage

class SessionProcessor:
    """
    Handles session processing with resilience for memory failures.
    """
    
    def __init__(self, max_retries: int = 3, cleanup_between_retries: bool = True, verbose: bool = True):
        """
        Initialize the session processor.
        
        Parameters:
        -----------
        max_retries : int
            Maximum number of retry attempts per session (default: 3)
        cleanup_between_retries : bool
            Whether to force garbage collection between retries (default: True)
        verbose : bool
            Whether to print detailed progress information (default: True)
        """
        self.max_retries = max_retries
        self.cleanup_between_retries = cleanup_between_retries
        self.verbose = verbose
        self.session_results = {}
        self.failed_sessions = {}
        self.retry_counts = {}
        
    def process_session_with_retries(self, session_id: str, process_func, *args, **kwargs) -> Tuple[bool, Optional[Any], Optional[str]]:
        """
        Process a single session with automatic retry logic.
        
        Parameters:
        -----------
        session_id : str
            Unique identifier for the session
        process_func : callable
            Function to process the session
        *args, **kwargs
            Arguments to pass to process_func
            
        Returns:
        --------
        Tuple[bool, Optional[Any], Optional[str]]
            (success, result, error_message)
        """
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                if self.verbose and attempt > 0:
                    print(f"    🔄 Retry {attempt}/{self.max_retries} for session {session_id}")
                
                # Monitor memory usage during processing
                with monitor_memory_usage(f"session {session_id} (attempt {attempt + 1})", 
                                         verbose=self.verbose) as monitor:
                    result = process_func(*args, **kwargs)
                
                # Success!
                if self.verbose:
                    if attempt > 0:
                        print(f"    ✅ Session {session_id} succeeded on retry {attempt}")
                    else:
                        print(f"    ✅ Session {session_id} succeeded on first attempt")
                
                return True, result, None
                
            except MemoryError as e:
                error_msg = f"Memory allocation failed: {str(e)}"
                
                if self.verbose:
                    print(f"    💾 Memory error in session {session_id} (attempt {attempt + 1}): {error_msg}")
                
                # Force cleanup between retries
                if attempt < self.max_retries and self.cleanup_between_retries:
                    if self.verbose:
                        print(f"    🧹 Forcing memory cleanup before retry...")
                    
                    # Aggressive cleanup
                    gc.collect()
                    force_garbage_collection(verbose=self.verbose)
                    
                    # Brief pause to let system settle
                    time.sleep(2)
                
                # If this was the last attempt, record the failure
                if attempt == self.max_retries:
                    if self.verbose:
                        print(f"    ❌ Session {session_id} failed after {self.max_retries + 1} attempts")
                    return False, None, error_msg
                    
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                
                if self.verbose:
                    print(f"    ❌ Unexpected error in session {session_id}: {error_msg}")
                    if self.verbose:  # Extra verbose for debugging
                        traceback.print_exc()
                
                # For non-memory errors, don't retry
                return False, None, error_msg
        
        # Should never reach here
        return False, None, "Maximum retries exceeded"
    
    def process_sessions_resilient(self, sessions: List[str], process_func, 
                                 session_args_func=None, common_kwargs=None) -> Dict:
        """
        Process multiple sessions with resilience.
        
        Parameters:
        -----------
        sessions : List[str]
            List of session identifiers
        process_func : callable
            Function to process each session
        session_args_func : callable, optional
            Function to generate session-specific arguments: session_id -> args
        common_kwargs : dict, optional
            Common keyword arguments for all sessions
            
        Returns:
        --------
        Dict
            Processing results with successful and failed sessions
        """
        if common_kwargs is None:
            common_kwargs = {}
        
        successful_sessions = {}
        failed_sessions = {}
        processing_summary = {
            'total_sessions': len(sessions),
            'attempted': 0,
            'successful': 0,
            'failed': 0,
            'retry_stats': {}
        }
        
        if self.verbose:
            print(f"🔄 Processing {len(sessions)} sessions with resilience...")
        
        for i, session_id in enumerate(sessions):
            processing_summary['attempted'] += 1
            
            if self.verbose:
                print(f"\n📊 Processing session {session_id} ({i+1}/{len(sessions)})")
            
            # Get session-specific arguments
            if session_args_func:
                session_args = session_args_func(session_id)
            else:
                session_args = (session_id,)
            
            # Process with retries
            success, result, error_msg = self.process_session_with_retries(
                session_id, process_func, *session_args, **common_kwargs
            )
            
            if success:
                successful_sessions[session_id] = result
                processing_summary['successful'] += 1
            else:
                failed_sessions[session_id] = error_msg
                processing_summary['failed'] += 1
            
            # Force cleanup after each session (regardless of success/failure)
            if self.cleanup_between_retries:
                force_garbage_collection(verbose=False)  # Silent cleanup
        
        # Final cleanup
        if self.verbose:
            print(f"\n🧹 Final memory cleanup...")
        force_garbage_collection(verbose=self.verbose)
        
        # Summary
        processing_summary['success_rate'] = (
            processing_summary['successful'] / processing_summary['total_sessions'] * 100
            if processing_summary['total_sessions'] > 0 else 0
        )
        
        if self.verbose:
            print(f"\n📊 Session processing summary:")
            print(f"  Total sessions: {processing_summary['total_sessions']}")
            print(f"  Successful: {processing_summary['successful']}")
            print(f"  Failed: {processing_summary['failed']}")
            print(f"  Success rate: {processing_summary['success_rate']:.1f}%")
            
            if failed_sessions:
                print(f"\n❌ Failed sessions:")
                for session_id, error in failed_sessions.items():
                    print(f"    {session_id}: {error}")
        
        return {
            'successful_sessions': successful_sessions,
            'failed_sessions': failed_sessions,
            'processing_summary': processing_summary
        }


def create_resilient_session_wrapper(original_process_func, verbose: bool = True):
    """
    Create a resilient wrapper around an existing session processing function.
    
    Parameters:
    -----------
    original_process_func : callable
        Original session processing function
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    callable
        Wrapped function with resilience
    """
    def resilient_wrapper(sessions: List[str], *args, **kwargs):
        processor = SessionProcessor(verbose=verbose)
        
        def session_args_func(session_id):
            # Extract session-specific arguments if needed
            return (session_id,) + args
        
        return processor.process_sessions_resilient(
            sessions=sessions,
            process_func=original_process_func,
            session_args_func=session_args_func,
            common_kwargs=kwargs
        )
    
    return resilient_wrapper


def process_with_memory_aware_batching(items: List[Any], process_func, 
                                     batch_size: int = 5, 
                                     memory_threshold_gb: float = 0.8,
                                     verbose: bool = True) -> Dict:
    """
    Process items in batches with memory monitoring and adaptive batch sizing.
    
    Parameters:
    -----------
    items : List[Any]
        Items to process
    process_func : callable
        Function to process each item
    batch_size : int
        Initial batch size (default: 5)
    memory_threshold_gb : float
        Memory threshold for reducing batch size (default: 0.8)
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    Dict
        Processing results
    """
    import psutil
    
    successful_items = {}
    failed_items = {}
    current_batch_size = batch_size
    
    if verbose:
        print(f"🔄 Processing {len(items)} items in adaptive batches...")
    
    for i in range(0, len(items), current_batch_size):
        batch = items[i:i + current_batch_size]
        batch_num = (i // current_batch_size) + 1
        
        if verbose:
            print(f"\n📦 Processing batch {batch_num} ({len(batch)} items)")
        
        # Check memory before batch
        memory_info = psutil.virtual_memory()
        available_gb = memory_info.available / (1024**3)
        
        if available_gb < memory_threshold_gb and current_batch_size > 1:
            current_batch_size = max(1, current_batch_size // 2)
            if verbose:
                print(f"  📉 Reduced batch size to {current_batch_size} (low memory: {available_gb:.1f} GB)")
            
            # Reprocess this batch with smaller size
            batch = items[i:i + current_batch_size]
        
        # Process batch with resilience
        processor = SessionProcessor(verbose=verbose)
        
        def batch_args_func(item_id):
            return (item_id,)
        
        batch_results = processor.process_sessions_resilient(
            sessions=[str(item) for item in batch],
            process_func=process_func,
            session_args_func=batch_args_func
        )
        
        # Aggregate results
        successful_items.update(batch_results['successful_sessions'])
        failed_items.update(batch_results['failed_sessions'])
        
        if verbose:
            print(f"  ✅ Batch {batch_num} complete: {len(batch_results['successful_sessions'])} success, "
                  f"{len(batch_results['failed_sessions'])} failed")
    
    return {
        'successful_items': successful_items,
        'failed_items': failed_items,
        'final_batch_size': current_batch_size
    }


if __name__ == "__main__":
    # Test the session resilience system
    import numpy as np
    
    def test_session_func(session_id: str, fail_probability: float = 0.3):
        """Test function that sometimes fails with memory errors."""
        print(f"    Processing session {session_id}")
        
        # Simulate memory-intensive operation
        if np.random.random() < fail_probability:
            raise MemoryError(f"Simulated memory allocation failure for session {session_id}")
        
        # Simulate successful processing
        time.sleep(0.5)
        return f"Result for {session_id}"
    
    # Test resilient processing
    sessions = [f"session_{i:03d}" for i in range(1, 11)]
    
    processor = SessionProcessor(max_retries=2, verbose=True)
    results = processor.process_sessions_resilient(
        sessions=sessions,
        process_func=test_session_func,
        session_args_func=lambda sid: (sid, 0.4),  # 40% failure rate
    )
    
    print("\n🎯 Test Results:")
    print(f"Successful: {len(results['successful_sessions'])}")
    print(f"Failed: {len(results['failed_sessions'])}")
    print(f"Success rate: {results['processing_summary']['success_rate']:.1f}%")