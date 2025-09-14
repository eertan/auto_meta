# AutoMeta: Automated Metadata Generation

![Project Status: WIP](https://img.shields.io/badge/status-work--in--progress-yellow.svg)

## Introduction

AutoMeta is a Python-based project designed to automatically generate metadata from a given set of data sources. It analyzes data tables from various formats (like CSV or Parquet) or database tables to infer important metadata, such as primary and foreign keys, and classifies each data source into a specific type. This automated process helps in understanding and organizing large datasets with minimal manual effort.

## Features

- **Automated Metadata Generation**: Automatically analyzes data sources and generates metadata.
- **Primary Key Detection**: Identifies potential primary keys by checking for uniqueness in columns with "id-looking" values.
- **Foreign Key Identification**: Discovers relationships between data sources by identifying foreign keys.
- **Data Source Classification**: Categorizes each data source into one of the following types:
    - **Entity**: Represents a business object with a primary key (e.g., User, Company).
    - **Event**: Records an action that occurred at a specific time, requiring a timestamp (e.g., Transaction).
    - **State**: Captures a state that is valid for a period, also requiring a timestamp (e.g., Weather, User Status).
    - **Relationship**: Links two or more entities together.
    - **Participation**: Connects an event to an entity.

## Getting Started

### Prerequisites

- Python 3.13 or higher
- [uv](httpss://github.com/astral-sh/uv) for dependency management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/auto_meta.git
    cd auto_meta
    ```
2.  **Create a virtual environment and install dependencies using uv:**
    ```bash
    uv venv
    uv pip sync
    ```

## Usage

To run the metadata generation process, execute the `main.py` script:

```bash
uv run python main.py
```

The script will perform the following steps:
1.  Load data from the `data/` directory.
2.  Initialize the classification agents.
3.  Build and run the classification graph.
4.  Output the final classifications for each data source.

## Roadmap

- [ ] Add support for more data source formats (e.g., Parquet, database connections).
- [ ] Improve the accuracy of primary and foreign key detection.
- [ ] Enhance the classification rules for more complex scenarios.
- [ ] Add a web interface for visualizing the generated metadata.

## License

This project is licensed under the terms of the LICENSE file.