#!/bin/bash

REQUIREMENTS_FILE="requirements.txt"

# Check if file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found!"
    exit 1
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting installation process..."

while read -r line || [ -n "$line" ]; do
    # Strip whitespace
    line=$(echo "$line" | xargs)

    # Skip empty lines and comments
    if [[ -z "$line" ]] || [[ "$line" == \#* ]]; then
        continue
    fi

    # Attempt Conda install first
    # We replace '==' with '=' because conda uses single equals for versioning
    conda_friendly_line=$(echo "$line" | sed 's/==/=/g')
    
    echo -e "${YELLOW}Attempting Conda install for: $line${NC}"
    
    # We use -c conda-forge because that is where fiona/gdal binaries live
    conda install --yes -c conda-forge "$conda_friendly_line"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed $line via Conda.${NC}"
    else
        echo -e "${RED}Conda failed for $line. Falling back to Pip...${NC}"
        
        # Fallback to Pip
        pip install "$line"
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully installed $line via Pip.${NC}"
        else
            echo -e "${RED}CRITICAL: Both Conda and Pip failed to install $line${NC}"
        fi
    fi
    echo "------------------------------------------------"

done < "$REQUIREMENTS_FILE"
```

### Usage

1.  Save the file as `install_deps.sh` in the same directory as your `requirements.txt`.
2.  Make it executable:
    ```bash
    chmod +x install_deps.sh
    ```
3.  Run it:
    ```bash
    ./install_deps.sh