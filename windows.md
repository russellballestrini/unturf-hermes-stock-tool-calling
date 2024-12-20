# Setting Up a Streamlit Python Application with Virtual Environment and Requirements.txt on Windows

This tutorial will guide you through the process of setting up a Streamlit Python application with a virtual environment and installing dependencies from a `requirements.txt` file on Windows. We will use the example project "https://github.com/russellballestrini/unturf-hermes-stock-tool-calling".

## Prerequisites

- Windows operating system
- Python installed (version 3.6 or higher)
- Git installed (if using the Git method)

## Step 1: Clone the Project Repository

### Option 1: Using Git

1. Open the Command Prompt by pressing `Win + R` and typing "cmd".
2. Type the following command and press Enter:
```
git clone https://github.com/russellballestrini/unturf-hermes-stock-tool-calling.git
```
3. The project repository will be cloned to your current directory.

### Option 2: Downloading the ZIP File

1. Open a web browser and navigate to the project's GitHub page: [https://github.com/russellballestrini/unturf-hermes-stock-tool-calling](https://github.com/russellballestrini/unturf-hermes-stock-tool-calling)
2. Click on the "Code" button and select "Download ZIP".
3. Save the downloaded ZIP file to your computer.

## Step 2: Navigate to the Project Directory

1. In the Command Prompt, type the following command and press Enter (if using the Git method):
```
cd unturf-hermes-stock-tool-calling
```
2. If you downloaded the ZIP file, extract the contents to a folder and navigate to that folder using the `cd` command. For example:
```
cd C:\Users\YourUsername\Desktop\unturf-hermes-stock-tool-calling
```
Replace "YourUsername" with your actual username.

## Step 3: Create a Virtual Environment

1. In the Command Prompt, type the following command and press Enter:
```
python -m venv env
```
Replace "env" with the desired name for your virtual environment.
2. A new folder named "env" will be created in your project directory.

## Step 4: Activate the Virtual Environment

1. In the Command Prompt, type the following command and press Enter:
```
env\Scripts\activate
```
Replace "env" with the name of your virtual environment.
2. You should see the name of your virtual environment in parentheses at the beginning of the Command Prompt, indicating that it is now active.

## Step 5: Install Dependencies from requirements.txt

1. In the Command Prompt, type the following command and press Enter:
```
pip install -r requirements.txt
```
This command will install all the packages listed in your `requirements.txt` file.
2. Wait for the installation process to complete. You should see a success message once it's finished.

## Step 6: Run the Streamlit Application

1. In the Command Prompt, type the following command and press Enter:
```
streamlit run app.py
```
This command will run the Streamlit application, and you should see the application in your default web browser.

## Step 7: Deactivate the Virtual Environment

1. Once you have finished working on your project, you can deactivate the virtual environment by typing the following command and pressing Enter:
```
deactivate
```
2. The virtual environment will be deactivated, and you will return to the standard Command Prompt.

## Conclusion

You have now successfully set up the "unturf-hermes-stock-tool-calling" Streamlit Python application with a virtual environment and installed dependencies from a `requirements.txt` file on Windows. This process helps keep your project's dependencies isolated and ensures that you have the correct versions of the required packages.

By following these steps, you can easily share your project with others and ensure that they have the same environment and dependencies to work on your project.
