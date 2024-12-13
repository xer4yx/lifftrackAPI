
![Logo](https://github.com/xer4yx/lifftrackAPI/blob/master/logo.jpg)

# LiftTrack API

LiftTrack API is a robust backend service designed for fitness enthusiasts and strength athletes who want to track their weightlifting progress. This API provides a comprehensive solution for managing workout data, including exercises, sets, reps, and personal records.

## Overview

This API enables users to:
- Track workout sessions and exercise performance
- Record and monitor personal records
- Manage exercise libraries and custom workouts
- Store and analyze lifting progress over time

Perfect for fitness applications, personal trainers, and individuals serious about their strength training journey. Built with modern best practices and secure authentication to protect user data.
## Features
### User Management
- User registration and authentication
- Secure login with JWT tokens
- User profile management
- Password reset functionality

### Workout Management
- Create, read, update, and delete workouts
- Track workout dates and durations
- Add notes to workout sessions
- Categorize workouts by type (e.g., strength, cardio)

### Exercise Tracking
- Log individual exercises within workouts
- Record sets, reps, and weights
- Track rest periods between sets
- Note exercise variations and modifications

### Progress Monitoring
- Track personal records (PRs)
- View historical workout data
- Monitor progress over time
- Calculate one-rep max (1RM)

### Data Management
- Store exercise library
- Custom exercise creation
- Workout templates
- Exercise categorization

### Security Features
- JWT-based authentication
- Password encryption
- Protected routes
- Secure data storage

### API Endpoints
- RESTful API design
- CRUD operations for all major features
- Pagination for large data sets
- Error handling and validation

## Sections
- [API References](https://github.com/xer4yx/lifftrackAPI/blob/master/APIREFS.md)
## Environment Variables

To run this project, you will need to add the following environment variables to your `.env` file

`FIREBASE_DB_URL`

`FIREBASE_AUTHENTICATION`

`FIREBASE_AUTH_UID`

`ROBOFLOW_API_URL`

`ROBOFLOW_API_KEY`

`ROBOFLOW_PROJECT_ID`

`ROBOFLOW_MODEL_VER`

`GOOGLE_SERVICES_JSON`


## Run Locally

Clone the project

```bash
  git clone https://github.com/xer4yx/lifftrackAPI
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the server

```bash
  uvicorn main:app --host 127.0.0.1 --port 8080
```


## Authors

- [@xer4yx](https://www.github.com/xer4yx)
- [@Twenty-2020](https://www.github.com/Twenty-2020)

