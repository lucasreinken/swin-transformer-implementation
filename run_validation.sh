#!/bin/bash
# Quick validation run script for cluster
# Usage: ./run_validation.sh

set -e  # Exit on error

echo "================================"
echo "CIFAR-100 Model Validation Setup"
echo "================================"
echo ""

# Step 1: Verify configuration
echo "Step 1: Verifying configuration..."
python3 validate_config.py
if [ $? -ne 0 ]; then
    echo "❌ Configuration validation failed!"
    exit 1
fi
echo "✅ Configuration valid"
echo ""

# Step 2: Check if container exists
echo "Step 2: Checking container..."
if [ ! -f "pml.sif" ]; then
    echo "⚠️  Container not found. Building..."
    echo "This will take 10-15 minutes..."
    echo "Requesting CPU node..."
    srun --partition=cpu-2h --pty bash -c 'apptainer build pml.sif pml.def'
    echo "✅ Container built"
else
    echo "✅ Container exists"
fi
echo ""

# Step 3: Check logs directory
echo "Step 3: Preparing directories..."
mkdir -p logs
mkdir -p runs
echo "✅ Directories ready"
echo ""

# Step 4: Verify CIFAR-100 is selected
echo "Step 4: Checking dataset selection..."
grep "DATASET = \"cifar100\"" config/__init__.py > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ CIFAR-100 not selected in config/__init__.py"
    exit 1
fi
echo "✅ CIFAR-100 selected"
echo ""

# Step 5: Verify validation is enabled
echo "Step 5: Checking validation mode..."
grep "\"enable_validation\": True" config/cifar100_config.py > /dev/null
if [ $? -ne 0 ]; then
    echo "❌ Validation not enabled in config/cifar100_config.py"
    exit 1
fi
echo "✅ Validation enabled"
echo ""

# Step 6: Show configuration
echo "================================"
echo "Current Configuration:"
echo "================================"
echo "Dataset: CIFAR-100"
echo "Variant: tiny"
echo "Pretrained model: swin_tiny_patch4_window7_224"
echo "Validation samples: 1000"
echo "Enable validation: True"
echo ""

# Step 7: Ask for confirmation
echo "Ready to run validation!"
echo ""
echo "Choose an option:"
echo "  1) Run interactive test (recommended first time)"
echo "  2) Submit batch job"
echo "  3) Exit"
echo ""
read -p "Enter choice [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "Starting interactive validation..."
        echo "This will take ~10-15 minutes"
        echo ""
        srun --partition=gpu-teaching-2h --gpus=1 --pty bash -c 'apptainer run --nv pml.sif python main.py'
        ;;
    2)
        echo ""
        echo "Submitting batch job..."
        JOBID=$(sbatch job.slurm | awk '{print $4}')
        echo "✅ Job submitted: $JOBID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u \$USER"
        echo "  tail -f logs/${JOBID}.out"
        echo "  scancel ${JOBID}  # to cancel if needed"
        echo ""
        ;;
    3)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "Next Steps:"
echo "================================"
echo "1. Check results in runs/run_XX/model_validation_results.json"
echo "2. Review training log in runs/run_XX/training.log"
echo "3. Compare Top-1 and Top-5 accuracies"
echo "4. If successful, disable validation and run full training"
echo ""
