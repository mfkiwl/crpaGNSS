<h1 align="center">
  charlizard
  
  <div align="center">
  
  ![GitHub Repo stars](https://img.shields.io/github/stars/sturdivant20/charlizard)
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
  ![GitHub pull requests](https://img.shields.io/github/issues-pr/sturdivant20/charlizard)
  ![GitHub issues](https://img.shields.io/github/issues/sturdivant20/charlizard)
  ![GitHub contributors](https://img.shields.io/github/contributors/sturdivant20/charlizard)
    
  </div>
  
</h1>

<h3 align="center">oh no, not again</h3>
<h4 align="center">Daniel Sturdivant &ltsturdivant20@gmail.com&gt</h4>

## Docs
- [Todo](#todo)
- [Poetry](#poetry)
- [Git Submodules](#git-submodules)

## Todo

## Poetry
To install poetry on MacOS or Linux:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```

```sh
poetry init     # create a new project based on current directory
poetry shell    # opens the virtual environment for current project
poetry install  # install all dependencies in pyproject.toml
poetry show     # show information about installed packages
```

## Git Submodules
Add a submodule with the following commands:
```sh
cd libs
git submodule add -b <branch> <git https url>
cd ..
```

Initialize module with:
```sh
git submodule init
```

For any submodules with submodules of their own, you must recursively initialize them:
```sh
git submodule update --init --recursive
```

To update submodules to latest commit, make sure commit is pushed and merge the commits:
```sh
git submodule update --remote --merge             # for submodules
git submodule update --remote --merge --recursive # for sub-submodules
```

## Symbolic Paths
To add sub-submodule to list of submodules, try the following command:
```sh
mkdir ./libs/<submodule>
sudo mount --bind ./libs/<submodule>/libs/<sub-submodule> ./libs/<sub-submodule>
```

To unmount the binded file (this is necessary the delete the repository locally):
```sh
sudo unmount ./libs/<sub-submoudle>
```
