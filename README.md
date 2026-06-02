# Tennis genetic algorithm

Implementation of a Tennis genetic algorithm with the usage of Gymnasium interface.

## Prerequisities (For Windows)
 - Download:
https://sourceforge.net/projects/vcxsrv/
 - Open XLauncher, options: default + disable access control, run the process. (Project only developed on Windows, this GUI should already work on Linux, you might change Dockerfile and docker-compose.yml in such case)

## Running the simulation
 - Run 'docker build -t tennis-app .' then 'docker compose up' to first run the simulation
 - After changing the code, rebuild with 'docker build --no-cache -t tennis-app .'

 ## TO-DO:
- [x] Create the repository
- [x] Learn more about the algorithm
- [x] Write the first report
- [x] Actually code
- [x] Create skeleton of the algorithm
- [x] Dummy fitness function

- [ ] Fitness function
      - [ ] Track metrics
      - [ ] Prepare pattern
      - [ ] Create chart of fitness values over time
      - [ ] Implement whole function

- [ ] Selection function
      - [] Research it more
      