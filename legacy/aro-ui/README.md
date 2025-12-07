
# AI-Powered Maritime Route Optimization with Risk Assessment and Iceberg Avoidance

## Overview
The **AI-Powered Maritime Route Optimization with Risk Assessment and Iceberg Avoidance** system uses artificial intelligence and machine learning to enhance Arctic navigation. The system optimizes maritime routes while considering iceberg locations, weather conditions, and oceanographic risks to ensure safe, efficient, and environmentally responsible shipping in the Arctic. The platform integrates predictive models, real-time hazard detection, and dynamic rerouting to improve Arctic shipping logistics.

The core technologies include **TensorFlow.js**, **Random Forest algorithms**, **K-Means clustering**, and **OpenLayers** for real-time data visualization. This project is intended to serve as a foundation for scalable and intelligent maritime navigation solutions, particularly in polar regions, with potential for global maritime applications.

## Key Features
- **Optimized Route Generation**: Generates safe sea-only routes between Arctic ports considering various environmental factors.
- **Hazard Alerts**: Provides real-time notifications for risks such as icebergs, storms, and shallow waters.
- **Dynamic Rerouting**: Automatically recalculates the route when new hazards are detected.
- **Risk Visualization**: Displays traffic data, iceberg density, and risk zones using interactive visualizations.
- **Real-time Risk Detection**: Classifies environmental and iceberg risks using machine learning models.
- **Port Traffic Monitoring**: Analyzes port traffic and clusters using K-Means clustering to identify optimal routing strategies.


---

### Frontend
1. **React 18 with TypeScript**:
    - React is the core library for building the user interface.
    - TypeScript is used for static typing in React.

2. **Tailwind CSS**:
    - Utility-first CSS framework to style the app efficiently.

3. **OpenLayers**:
    - For handling map visualizations.

4. **Lucide React**:
    - For interactive icons in the UI.

5. **Zustand**:
    - A simple state management library.

6. **Recharts**:
    - For data visualization, especially for analytics.

---

### Backend & AI

1. **Node.js with Express**:
    - The backend API to handle requests from the frontend.

2. **Python (Flask/FastAPI)**:
    - Python used for integrating AI models.

3. **TensorFlow.js**:
    - Client-side AI processing for running machine learning models directly in the browser.

4. **Redis**:
    - A tool for real-time event handling, which can be used for caching, message brokering, etc.

---

### Machine Learning Models
1. **K-means clustering**:
    - Used for analyzing traffic patterns.

---

### `requirements.txt` for Backend (Python-related libraries)

```txt
Flask==2.2.2
FastAPI==0.78.0
uvicorn==0.18.2
tensorflow==2.11.0
scikit-learn==1.1.2
pandas==1.5.0
numpy==1.23.4
redis==4.4.0
```

### `package.json` for Frontend (JavaScript libraries)

Here’s how you would set up your **`package.json`** for the frontend dependencies:

```json
{
  "name": "maritime-navigation-app",
  "version": "1.0.0",
  "private": true,
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "dependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0",
    "react-scripts": "4.0.3",
    "typescript": "^4.4.0",
    "tailwindcss": "^2.0.0",
    "openlayers": "^6.0.0",
    "lucide-react": "^0.0.1",
    "zustand": "^3.0.0",
    "recharts": "^2.0.0"
  },
  "devDependencies": {
    "postcss": "^8.0.0",
    "autoprefixer": "^10.0.0"
  }
}
```

---

### How to use the `requirements.txt`
For **backend**:

1. Install all necessary Python libraries as per the `requirements.txt` file using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

2. If you need to generate `requirements.txt` based on your current Python environment, you can run:
    ```bash
    pip freeze > requirements.txt
    ```

For **frontend**

1. Initialize your React app with TypeScript by running the following:
    ```bash
    npx create-react-app maritime-navigation-app --template typescript
    ```

2. Install the necessary frontend libraries:
    ```bash
    npm install react react-dom react-scripts tailwindcss openlayers lucide-react zustand recharts
    ```

3. To build the React app, run:
    ```bash
    npm run build
    ```

---

## Prerequisites

Before running the system, ensure you have the following tools installed:

- **Node.js**: v18+ (for React.js, TensorFlow.js, and Zustand)
- **Python**: v3.10 or higher (for machine learning models)
- **Git**: For version control.

### Required Node.js Modules

1. **React**: For building the frontend web app.
2. **TensorFlow.js**: For machine learning model inference and prediction.
3. **Zustand**: For managing global state in the frontend.
4. **OpenLayers**: For mapping and visualizing navigational routes.
5. **Recharts**: For visualizing risks and traffic data.
6. **Axios**: For making HTTP requests to external APIs.

To install these dependencies, run the following command in the root directory:

```bash
npm install
```

### Required Python Libraries

1. **scikit-learn**:- For implementing machine learning models like Random Forest and K-Means clustering.
2. **Pandas**:- For data manipulation and handling.
3. **Numpy**:- For numerical computations.
4. **Flask or FastAPI**:- For creating the backend API server (if needed).

To install the required Python libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Clone the Repository

To get started with the project, clone the repository to your local machine using Git:

```bash
git clone https://github.com/Yash22222/Arctic-Route-Optimization-System.git
```

Navigate to the project directory:

```bash
cd maritime-route-optimization
```

## Setup and Configuration

### 1. Set Up the Frontend

#### Step 1: Install Node.js Dependencies
Run the following command to install all necessary dependencies for the frontend:

```bash
npm install
```

#### Step 2: Start the Development Server

Start the React development server to run the web application locally:

```bash
npm start
```

The app should now be running at `http://localhost:3000`.

#### Step 3: Access the Application
Open a browser and go to `http://localhost:3000` to access the interactive map and dashboard where routes and risks are displayed in real time.

### 2. Set Up the Backend

#### Step 1: Install Python Dependencies
Ensure you have installed Python 3.10 or higher. Then, install the necessary Python packages:

```bash
pip install -r requirements.txt
```

The backend server should now be running, typically at `http://localhost:5000`. You can integrate APIs or handle the machine learning logic through this server.


## How to Contribute

If you would like to contribute to this project, follow these steps:

1. Fork the repository by clicking the "Fork" button at the top of the repository page.
2. Clone your forked repository:

```bash
git clone https://github.com/Yash22222/Arctic-Route-Optimization-System.git
```

3. Create a new branch for your feature:

```bash
git checkout -b feature-xyz
```

4. Make your changes and commit them:

```bash
git add .
git commit -m "Add new feature XYZ"
```

5. Push your changes:

```bash
git push origin feature-xyz
```

6. Open a pull request to the `main` branch.

### Code Style Guidelines

- Follow **PEP 8** for Python code style.
- Use **ESLint** for JavaScript/TypeScript linting and follow **Airbnb** style guide for React code.
- Write unit tests for new features and bug fixes.

## Future Work
1. **Satellite Remote Sensing Data Integration**: Integrating real-time satellite data sources such as Sentinel-1 (SAR) and MODIS to enhance route accuracy and iceberg detection.
2. **3D Oceanographic and Weather Modeling**: Introducing 3D modeling to simulate ocean currents, wind profiles, and underwater topography for more accurate predictions.
3. **Reinforcement Learning for Autonomous Navigation**: Implementing RL agents for adaptive route planning based on evolving Arctic conditions and historical data.
4. **Vessel-Specific Optimization**: Enhancing the platform to consider vessel parameters like size, draft, speed, and fuel consumption for customized route planning.
5. **User Feedback Loop**: Introducing a feedback system to improve model performance based on real-world user inputs.
6. **Global Maritime Route Support**: Expanding the system to support global sea routes, including tropical cyclone zones, piracy-prone areas, and coral reefs.
7. **Blockchain Integration**: Utilizing blockchain technology for secure logging of voyage data, ensuring data integrity, and compliance with maritime regulations.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For questions or inquiries, feel free to contact the project maintainers:

- **Project Lead**:- [Yash Shirsath](mailto:yashshirsath2410@gmail.com)
- **GitHub**:- [Yash22222](https://github.com/Yash22222)

