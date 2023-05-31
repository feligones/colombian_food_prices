# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Install project dependencies using Pipenv
RUN pipenv install --deploy --system

# Copy the application code into the container
COPY app ./

# Copy the .env file into the container
COPY .env ./

# Set the entry point command to start the Dash application
CMD ["pipenv", "run", "gunicorn", "dash_plot:app", "-b", "0.0.0.0:8050"]