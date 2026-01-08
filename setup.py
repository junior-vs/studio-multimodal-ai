from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="multimodal-ai-project",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A modular multimodal AI project for images, videos, and text analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/multimodal-ai-project",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.3.0",
        "scikit-image>=0.18.0",
        "moviepy>=1.0.3",
        "imageio>=2.9.0",
        "nltk>=3.6.0",
        "spacy>=3.4.0",
        "transformers>=4.20.0",
        "scikit-learn>=1.0.0",
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "tensorflow>=2.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "notebook>=6.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
        ],
    },
)
