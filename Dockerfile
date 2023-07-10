# Use a Python base image compatible with Elastic Beanstalk
FROM python:3.10-slim-buster

# Set the working directory
WORKDIR /app

# Copy the Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Install Pipenv
RUN pip install --no-cache-dir pipenv

# Install project dependencies using Pipenv
RUN pipenv install --deploy --system --ignore-pipfile

# Copy the application code into the container
COPY app ./

# Copy the .env file into the container (optional)
COPY .env ./

EXPOSE 8050

# Set the entry point command to start the Dash application
CMD ["pipenv", "run", "gunicorn", "dash_plot:app", "-b", "0.0.0.0:8050"]
