# AI-Enabled-Closed-loop-Insulin-System

This project uses a Long Short-Term Memory (LSTM) neural network to forecast future blood glucose levels based on historical data. It is built with PyTorch and trained on the Ohio T1DM Dataset.

The core of this project is a sequence-to-sequence model that takes a 3-hour window of historical data (at 5-minute intervals) and predicts the physiological data for the next 5-minute timestep. This type of forecasting is a critical component for building AI-enabled closed-loop insulin delivery systems (an "artificial pancreas").

## 📈 Project Features

  * **LSTM Model:** A multi-layer LSTM network built with PyTorch for time-series forecasting.
  * **Data Preprocessing:** A robust pipeline for the Ohio T1DM dataset, including:
      * Cubic Spline interpolation for missing Continuous Glucose Monitoring (CGM) data.
      * `MinMaxScaler` normalization for all features.
  * **Custom DataLoaders:** Efficient PyTorch `Dataset` and `DataLoader` for handling the sequence data from multiple patient files.
  * **Resumable Training:** The training script automatically saves checkpoints (`training_checkpoint.pth`) and can be stopped and resumed at any time.
  * **Best Model Saving:** The script saves the model with the lowest validation loss to `best_model.pth`.
  * **Evaluation:** Automatically calculates and displays the **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)** for glucose prediction on the test set.

## 🛠️ Setup and Installation

### 1\. Requirements

You can install all necessary Python packages using `pip`.

```bash
pip install torch pandas numpy scipy scikit-learn matplotlib
```

Alternatively, you can create a `requirements.txt` file with the following content and run `pip install -r requirements.txt`:

**requirements.txt**

```
torch
pandas
numpy
scipy
scikit-learn
matplotlib
```

### 2\. Get the Data

This model is designed to work with the **Ohio T1DM Dataset** (from 2018 and 2020).

1.  **Download the data:** You must acquire the dataset yourself. The script assumes it is in a zip file named `Ohio Data.zip`.
2.  **File Structure:** The `Ohio Data.zip` file should contain the processed data in the following structure:
    ```
    Ohio Data/
    ├── Ohio2018_processed/
    │   ├── train/
    │   │   ├── 540-ws-training.csv
    │   │   ├── ...
    │   └── test/
    │       ├── 540-ws-testing.csv
    │       ├── ...
    └── Ohio2020_processed/
        ├── train/
        │   ├── 544-ws-training.csv
        │   ├── ...
        └── test/
            ├── 544-ws-testing.csv
            ├── ...
    ```

## 🚀 How to Run

This project was originally a Google Colab notebook but can be run locally with one minor change.

### Option 1: Running Locally (Recommended)

1.  **Clone the repository:**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Place your data:** Put the `Ohio Data.zip` file in the root of the project directory.

3.  **IMPORTANT: Edit the script:**
    Open the Python script (`.py` or `.ipynb` file) and find this line:

    ```python
    # IMPORTANT: Verify this path matches your project folder in Google Drive.
    # It should contain the 'Ohio Data.zip' file.
    HOME_DIR = 'drive/MyDrive/vinay_glucose'
    ```

    **Change it** to point to your current directory:

    ```python
    # Path to the project root directory
    HOME_DIR = '.' 
    ```

4.  **Run the script:**
    Execute the Python script.

    ```bash
    python your_script_name.py
    ```

    The script will automatically:

      * Unzip `Ohio Data.zip` into an `Ohio Data/` folder (if it doesn't already exist).
      * Create the DataLoaders.
      * Build the LSTM model.
      * Start (or resume) training for 150 epochs.
      * Save `training_checkpoint.pth` after each epoch.
      * Save `best_model.pth` whenever a new best validation loss is achieved.
      * After training, load the `best_model.pth` and print the final RMSE and MAE metrics on the test set.

### Option 2: Running on Google Colab

1.  **Upload to Google Drive:**

      * Create a folder in your Google Drive (e.g., `vinay_glucose`).
      * Upload your Jupyter Notebook (`.ipynb`) and the `Ohio Data.zip` file to this folder.

2.  **Open in Colab:**

      * Open the notebook from your Google Drive.

3.  **Set Runtime:**

      * Go to **Runtime -\> Change runtime type**.
      * Select **T4 GPU** (or any available GPU) from the "Hardware accelerator" dropdown.

4.  **Run All Cells:**

      * Click **Runtime -\> Run all**.
      * You will be prompted to mount your Google Drive.
      * The script will run exactly as described in the local setup, saving checkpoints and the best model directly to your Google Drive folder.

 🔬 Code Explanation

  * **`get_scaler(data_df)`:** A helper function that takes a dataframe, interpolates missing glucose values, fills other NaNs, and fits a `MinMaxScaler` that is used for all data.
  * **`OhioT1DMDataset(Dataset)`:** The custom PyTorch `Dataset` class. It loads all patient CSVs, preprocesses them using the global scaler, and serves up input/target sequences of length `SEQ_LENGTH` (36 steps, or 3 hours).
  * **`create_dataloader(...)`:** A wrapper function to create the `DataLoader` instances for training and testing.
  * **`SimpleLSTM(nn.Module)`:** The PyTorch model definition, consisting of an `nn.LSTM` layer followed by an `nn.Linear` layer.
  * **`train(...)`:** The main training function. It handles the training loop, validation loop, optimizer steps, learning rate scheduling, and checkpoint saving.
  * **`compute_and_display_metrics(...)`:** The evaluation function. It loads the best model, runs inference on the test set, inverse-transforms the scaled predictions, and calculates the final error metrics for the `cbg` (glucose) column.
  * **Main Execution Block:** The final part of the script sets hyperparameters (like `SEQ_LENGTH`, `BATCH_SIZE`, `NUM_EPOCHS`) and executes the full data loading, training, and evaluation pipeline.

 📊 Results

After training is complete, the script will print the final performance metrics on the test set. The output will look similar to this:

```
========================================
 Model Performance Metrics (Test Set)
========================================
Metric      | Glucose (mg/dL)
----------------------------------------
RMSE        | 21.34
MAE         | 15.12
========================================
```

*(Note: Your exact values may vary.)*
