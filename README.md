# Enzyme Evolution Deep Learning Pipeline
A comprehensive machine learning framework for enzyme evolution using both Multi-Layer Perceptron (MLP) and Transformer architectures to predict enzyme activity from sequence data and design improved variants.

## Overview
This pipeline implements a complete workflow for enzyme evolution using deep learning. The system features controlled mutation generation from 0-6 site mutations, dual model architecture with MLP and Transformer approaches, comprehensive evaluation and comparison tools, automated model saving with metadata, and advanced visualization and analysis capabilities.

## Installation
The pipeline requires Python 3.7+ with PyTorch 1.9+, NumPy 1.19+, Pandas 1.3+, Scikit-learn 1.0+, Matplotlib 3.3+, Seaborn 0.11+, and SciPy 1.7+. Install dependencies using pip install for torch, torchvision, torchaudio, numpy, pandas, scikit-learn, matplotlib, seaborn, and scipy. A multi-core CPU with 8GB+ RAM is recommended, with optional CUDA-compatible GPU for faster training.

## Quick Start
Download the script and run python enzyme_evolution.py to execute the complete pipeline. The script automatically generates synthetic enzyme variant data, trains both MLP and Transformer models, evaluates and compares performance, designs new enzyme variants, saves models with metadata, and creates comprehensive visualizations.

## Usage
Basic usage involves importing the EnzymeEvolutionPipeline class, defining your enzyme and substrate sequences, initializing the pipeline, generating training data, preparing datasets, training both models, and designing variants. Advanced configuration allows customization of mutation library composition, model parameters including batch size and learning rate, and prediction of activities for new sequences using either model type.

## Model Architectures
The MLP model uses an enhanced multi-layer perceptron with one-hot encoded amino acid sequences, batch normalization for stable training, dropout for regularization, and mutation-level aware inputs. The architecture includes hidden layers of sizes [512, 256, 128, 64, 32] with ReLU activation and progressive dropout rates.
The Transformer model employs multi-head attention mechanism with 8 heads, positional encoding for sequence position awareness, 4 encoder layers with 512-dimensional feedforward networks, and global pooling for final prediction. The model processes enzyme and substrate sequences jointly with a model dimension of 128.

## Data Format
Input sequences should use standard amino acid letters, with substrate sequences representing FRET pair substrates and activity values as continuous fluorescence signals where 0 indicates no activity and higher values indicate stronger activity. The generated data structure includes sequences, activities, and mutation levels indicating the number of mutations from wild-type.

## Output Files
The system creates a saved_models directory containing model weight files with timestamps, metadata JSON files with performance metrics, and temporary best model files during training. Metadata includes model type, timestamp, comprehensive metrics like MSE and R-squared values, and the full model path for easy loading.

## Customization
Users can implement custom activity functions by subclassing EnzymeDataGenerator and overriding the simulate_activity method. Custom model architectures can be created by extending the base PyTorch modules. Real experimental data can be integrated by replacing synthetic data generation with CSV loading functions and calculating mutation levels from actual sequences.

## API Reference
The EnzymeEvolutionPipeline class provides methods for generating training data with controlled mutation levels, preparing train/validation/test datasets, training both MLP and Transformer models with customizable parameters, evaluating model performance with comprehensive metrics, predicting activities for new sequences, and designing improved variants. The EnzymeDataGenerator handles mutation generation and activity simulation, while ModelManager provides saving and loading functionality with metadata tracking.

## Troubleshooting
Common issues include CUDA out of memory errors, which can be resolved by reducing batch size, and slow training, which can be addressed by reducing model complexity or dataset size. Poor model performance often indicates data quality issues or need for hyperparameter adjustment. Memory issues with large sequences can be handled by processing in chunks rather than full batches.
Performance optimization involves using GPU acceleration when available, starting with batch sizes of 32 for MLP and 16 for Transformer, setting learning rates of 0.001 for MLP and 0.0005 for Transformer, utilizing built-in early stopping to prevent overfitting, and ensuring balanced representation across all mutation levels in the training data.

## Expected Results
Typical performance benchmarks show MLP models achieving R-squared values above 0.85 with Pearson correlations exceeding 0.90, while Transformer models generally achieve R-squared above 0.87 with Pearson correlations over 0.92. Training time ranges from 5-15 minutes depending on hardware configuration. Mutation analysis reveals clear activity trends versus mutation level, model comparison typically shows Transformer outperforming MLP, and variant design produces top variants with predicted activity improvements of 10-30%.

## Best Practices
Data preparation should ensure clean, valid amino acid sequences with normalized activity ranges and diverse mutation patterns, always using held-out test sets for validation. Model training benefits from starting with default hyperparameters before optimization, using multiple random seeds for robust results, monitoring validation curves to prevent overfitting, and considering ensemble approaches that combine MLP and Transformer predictions.
Variant design should begin with conservative approaches using fewer mutations, always validate predictions experimentally, use experimental results iteratively to improve models, and design diverse variant sets rather than focusing solely on top predictions. This iterative approach ensures practical success in real enzyme engineering applications.

## Citation
If you use this pipeline in your research, please cite: "Junming (Jimmy) Lao. AI4LifeScience-DeepLearning-for-Enzyme-variant-screening-High-throughput-Microfluidics"

## License and Support
This project is licensed under the MIT License. For questions or issues, check the troubleshooting section, open a GitHub issue, or contact the maintainers. Contributions are welcome for new model architectures, additional analysis tools, performance optimizations, bug fixes, and documentation improvements.
