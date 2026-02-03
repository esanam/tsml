# Time Series Deep Learning with PyTorch & tsai

A comprehensive starter project for learning modern deep learning techniques for time series analysis using PyTorch and the fastai/tsai ecosystem.

## 🎯 What You'll Learn

- Time series data preparation and windowing techniques
- Building LSTM, CNN, and Transformer models from scratch
- Using tsai for high-level, fastai-style APIs
- Training, evaluation, and visualization best practices
- Modern architectures and state-of-the-art techniques

## 🚀 Quick Start

### 1. Installation

```bash
# Create a virtual environment (recommended)
python -m venv ts_env
source ts_env/bin/activate  # On Windows: ts_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Jupyter Notebook

```bash
jupyter notebook time_series_deep_learning_starter.ipynb
```

### 3. Run the Cells

Start from the top and run each cell sequentially. The notebook is designed to be self-contained with synthetic data, so you can run it immediately without downloading datasets.

## 📚 Notebook Structure

1. **Setup & Installation** - Get your environment ready
2. **Data Generation** - Create synthetic time series data
3. **Dataset Preparation** - Learn windowing and splitting techniques
4. **LSTM Model** - Classic recurrent architecture
5. **Training Loop** - Standard PyTorch training patterns
6. **CNN Model** - Convolutional networks for time series
7. **tsai Integration** - High-level API usage
8. **Evaluation** - Metrics and visualization
9. **Transformer Model** - Attention-based architecture
10. **Next Steps** - Resources and advanced topics
11. **Exercises** - Practice problems

## 🎓 Learning Path

### Beginner (Start Here)
1. Run the notebook with synthetic data
2. Understand the windowing concept
3. Train the LSTM model
4. Experiment with hyperparameters

### Intermediate
1. Load your own CSV data
2. Try the CNN and Transformer architectures
3. Compare model performances
4. Implement multi-variate forecasting

### Advanced
1. Use tsai for state-of-the-art models (InceptionTime, TST)
2. Implement probabilistic forecasting
3. Add attention visualization
4. Build ensemble models

## 📊 Real-World Datasets to Try

Once comfortable with the basics, try these datasets:

- **Yahoo Finance**: Stock price data via `yfinance` library
- **UCI Machine Learning Repository**: Various time series datasets
- **Kaggle**: Store sales, web traffic, energy consumption
- **OpenWeatherMap API**: Weather forecasting data
- **Your own data**: CSV files with timestamp and values

## 🛠️ Key Libraries

- **PyTorch**: Core deep learning framework
- **fastai**: High-level training APIs
- **tsai**: Time series extension of fastai
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

## 📖 Additional Resources

### Documentation
- [tsai Documentation](https://timeseriesai.github.io/tsai/)
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/)
- [fastai Course](https://course.fast.ai/)

### Papers & Architectures
- **N-BEATS**: Neural basis expansion analysis for interpretable time series forecasting
- **Temporal Fusion Transformers**: Multi-horizon time series forecasting
- **Informer**: Efficient transformer for long sequence time-series forecasting
- **InceptionTime**: Finding AlexNet for time series classification

### Alternative Libraries
- **Darts**: User-friendly forecasting (by Unit8)
- **GluonTS**: Probabilistic models (by Amazon)
- **Prophet**: Simple forecasting (by Facebook)
- **statsforecast**: Statistical + ML models

## 💡 Tips for Success

1. **Start simple**: Begin with LSTM on synthetic data before trying complex models
2. **Understand your data**: Plot it, check for trends, seasonality, outliers
3. **Proper splitting**: Never shuffle time series - use temporal splits
4. **Scale your data**: Normalization is crucial for neural networks
5. **Iterate quickly**: Start with small models and short training times
6. **Validate properly**: Use a separate validation set, not just train/test
7. **Monitor overfitting**: Watch for divergence between train and validation loss

## 🔧 Troubleshooting

### CUDA/GPU Issues
If you get CUDA errors, add this at the start:
```python
device = torch.device('cpu')  # Force CPU usage
```

### Memory Errors
Reduce batch size or sequence length:
```python
batch_size = 16  # Instead of 32
window_size = 30  # Instead of 50
```

### Import Errors
Make sure all packages are installed:
```bash
pip install --upgrade -r requirements.txt
```

## 🎯 Next Steps After This Tutorial

1. **Complete the exercises** in the notebook
2. **Load your own data** and adapt the code
3. **Explore tsai models**: InceptionTime, XceptionTime, TST
4. **Try PyTorch Forecasting**: Temporal Fusion Transformer, DeepAR
5. **Read papers**: Understand the theory behind modern architectures
6. **Join communities**: fastai forums, PyTorch discussions, r/MachineLearning

## 🤝 Contributing

Found an error or have suggestions? Feel free to:
- Experiment with the code
- Try different architectures
- Test on new datasets
- Share your findings!

## 📝 License

This educational material is provided for learning purposes. Feel free to use and modify for your projects.

---

**Happy Learning! 🚀**

Remember: The best way to learn is by doing. Don't just read the code - run it, break it, fix it, and make it your own!
