#!/bin/bash

# Script to create a Python virtual environment, activate it, install requirements, and install a kernel for Jupyter/IPython.

# --- Configuration ---
VENV_NAME=".venv"  # Name of the virtual environment (default: .venv)
REQUIREMENTS_FILE="requirements.txt" # Path to the requirements file (default: requirements.txt)
KERNEL_NAME=".venv_kernel" # Name of the kernel to be displayed in Jupyter (default: .venv_kernel)

# --- Functions ---

create_venv() {
  echo "Creating virtual environment: $VENV_NAME"
  if [[ -d "$VENV_NAME" ]]; then
    echo "Virtual environment '$VENV_NAME' already exists."
  else
    python3 -m venv "$VENV_NAME"
    if [[ $? -ne 0 ]]; then
      echo "Error: Failed to create virtual environment '$VENV_NAME'."
      exit 1
    fi
    echo "Virtual environment '$VENV_NAME' created successfully."
  fi
}

activate_venv() {
  echo "Activating virtual environment: $VENV_NAME"
  source "$VENV_NAME/bin/activate"
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to activate virtual environment '$VENV_NAME'."
    exit 1
  fi
  echo "Virtual environment '$VENV_NAME' activated."
}

install_requirements() {
  echo "Installing requirements from: $REQUIREMENTS_FILE"
  if [[ -f "$REQUIREMENTS_FILE" ]]; then
    pip install --upgrade pip
    pip install -r "$REQUIREMENTS_FILE"
    if [[ $? -ne 0 ]]; then
      echo "Error: Failed to install requirements from '$REQUIREMENTS_FILE'."
      exit 1
    fi
    echo "Requirements installed successfully."
  else
    echo "Warning: Requirements file '$REQUIREMENTS_FILE' not found. Skipping installation."
  fi
}

install_kernel() {
  echo "Installing kernel: $KERNEL_NAME"
  python -m ipykernel install --user --name="$VENV_NAME" --display-name="$KERNEL_NAME"
  if [[ $? -ne 0 ]]; then
    echo "Error: Failed to install kernel '$KERNEL_NAME'."
    exit 1
  fi
  echo "Kernel '$KERNEL_NAME' installed successfully."
}

# --- Main Script ---

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "Error: python3 could not be found. Please make sure it is installed and in your PATH."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null
then
    echo "Error: pip could not be found. Please make sure it is installed and in your PATH."
    exit 1
fi

# Check if ipykernel is available
if ! command -v python -c "import ipykernel" &> /dev/null
then
    echo "Warning: ipykernel not found. Installing it globally."
    pip install ipykernel
    if [[ $? -ne 0 ]]; then
        echo "Error: Failed to install ipykernel."
        exit 1
    fi
fi

# Create the virtual environment
create_venv

# Activate the virtual environment
activate_venv

# Install requirements
install_requirements

# Install the kernel
install_kernel

echo "Setup complete. You can now use the '$KERNEL_NAME' kernel in your Jupyter notebooks."

# Deactivate the virtual environment (optional, but good practice)
deactivate

echo "Virtual environment deactivated."
