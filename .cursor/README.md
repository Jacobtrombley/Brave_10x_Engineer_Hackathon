# Cursor AI Configuration

This folder contains configuration settings and rules for Cursor AI's behavior within the Job Matching Recommendation System project.

## Folder Structure

- **settings.json**: Global AI behavior configuration for the entire project
- **rules/**: Directory containing specialized rule sets for different components
  - **ui.json**: Rules for UI components and styling
  - **models.json**: Rules for machine learning models and algorithms
  - **data.json**: Rules for data pipelines and processing

## Purpose of Project Rules

Project rules in Cursor AI enable granular control over AI behavior within different parts of a codebase. These rules help maintain consistency, enforce best practices, and customize AI assistance based on specific requirements of different components.

## How Rules Are Applied

1. Rules are matched based on file patterns specified in each rule set
2. More specific patterns take precedence over general patterns
3. When working in a file, Cursor AI applies the most relevant rules based on the file path
4. Global rules from `settings.json` apply to all files unless overridden by specific rules

## Rules Configuration

Each rule set contains:

- **name**: Descriptive name for the rule set
- **description**: Purpose of the rules
- **include**: File patterns where rules should be applied
- **rules**: Array of specific rules with:
  - **description**: What the rule is for
  - **pattern**: Specific file pattern for the rule
  - **ai**: AI behavior configuration
    - **allow**: Permitted AI operations
    - **guidelines**: Specific instructions for AI when working with matching files

For detailed documentation on how Cursor AI was used in this project, please see [CURSOR.md](../CURSOR.md) in the project root.
