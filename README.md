# LiftTrack API

![Logo](logo.jpg)

**Version:** `v1.10.4` *(Auto-loaded from `version.py`)*

LiftTrack API is a powerful backend service designed for fitness enthusiasts and strength athletes to effectively track their weightlifting progress. It provides endpoints to manage workouts, exercises, performance logs, and personal records with modern best practices and robust security features.

---

## 🚀 Features

### 👤 User Management
- Secure user registration and login
- Profile management
- User data management support

### 🏋️ Weightlifting Management
- CRUD and WebSocket operations for weightlifting
- Log dates, durations, and weightlifting data
- Workout categorization (e.g., Upper Body, Lower Body)

### 📝 Exercise Tracking
- Record suggestions, and body angles
- Track rest periods and variations
- Detailed logging for each workout

### 📈 Progress Monitoring
- View historical performance data

### 🔐 Security
- Token-based authentication
- Encrypted passwords
- Protected API routes
- Secure user data storage

---

## 🛠️ Getting Started (Development)

### 1. Clone the Repository

```bash
git clone https://github.com/xer4yx/lifttrackAPI
cd lifttrackAPI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Development Server

```bash
uvicorn main:app --host 127.0.0.1 --port 8080
```

---

## 👥 Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository

2. Create your feature branch (git checkout -b feature/awesome-feature)

3. Commit your changes (git commit -m 'Add awesome feature')

4. Push to the branch (git push origin feature/awesome-feature)

5. Open a Pull Request

Please follow our coding standards and write tests for any new features or bug fixes.

---

## Authors

- [@xer4yx](https://www.github.com/xer4yx)
- [@Twenty-2020](https://www.github.com/Twenty-2020)

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.