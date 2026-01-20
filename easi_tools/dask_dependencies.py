
def setup_cluster(client):
    """
    One-stop setup for all cluster dependencies
    """
    def setup_all_dependencies():
        import subprocess
        import sys
        import shutil
        from pathlib import Path
        
        # Install external packages
        packages = [
            'dask-pytorch-ddp',
            'zarr>=3.0.8',
            'icechunk',
            'nvidia-dali-cuda120'
        ]

        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                *packages, '--quiet'
            ])
        except Exception as e:
            return f"FAILED - Packages: {e}"

        # Setup easi_tools
        clone_dir = Path("/tmp/easi-notebooks")

        if clone_dir.exists():
            shutil.rmtree(clone_dir)

        # Clear cache
        for mod in [k for k in sys.modules.keys() if k.startswith('easi_tools')]:
            del sys.modules[mod]

        # Clone
        try:
            subprocess.run([
                "git", "clone",
                "https://github.com/cg379/easi-notebooks.git",
                str(clone_dir)
            ], check=True, capture_output=True, timeout=60)
        except Exception as e:
            return f"FAILED - Clone: {e}"

        if str(clone_dir) not in sys.path:
            sys.path.insert(0, str(clone_dir))

        # Verify
        try:
            from easi_tools.dask_helpers import load_config
            return "SUCCESS - Complete"
        except ImportError as e:
            return f"FAILED - easi_tools: {e}"

    # Run on workers and scheduler
    print("Setting up cluster (workers + scheduler)...")
    worker_results = client.run(setup_all_dependencies)
    scheduler_result = client.run_on_scheduler(setup_all_dependencies)
    
    # Check results
    all_success = all("SUCCESS" in str(r) for r in worker_results.values())
    all_success &= "SUCCESS" in str(scheduler_result)
    
    if all_success:
        print("Cluster setup complete!")
    else:
        print("WARNING: Some workers/scheduler failed:")
        print("Workers:", worker_results)
        print("Scheduler:", scheduler_result)
    
    return all_success