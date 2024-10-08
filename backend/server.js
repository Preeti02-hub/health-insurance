const express = require("express");
const cors = require("cors");
const { PythonShell } = require("python-shell");
const path = require("path");

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Prediction endpoint
app.post("/api/predict", (req, res) => {
	const { age, sex, bmi, children, smoker, region } = req.body;

	let options = {
		pythonPath: "python",
		scriptPath: path.join(__dirname, "models"),
		args: [JSON.stringify({ age, sex, bmi, children, smoker, region })],
	};

	PythonShell.run("ml_model.py", options)
		.then((messages) => {
			const prediction = parseFloat(messages[messages.length - 1]);
			res.json({ prediction });
		})
		.catch((err) => {
			console.error("Error:", err);
			res.status(500).json({ error: "Prediction failed" });
		});
});

// Training endpoint
app.post("/api/train", (req, res) => {
	let options = {
		pythonPath: "python",
		scriptPath: path.join(__dirname, "models"),
	};

	PythonShell.run("ml_model.py", options)
		.then((messages) => {
			res.json({ message: "Model training completed" });
		})
		.catch((err) => {
			console.error("Error:", err);
			res.status(500).json({ error: "Training failed" });
		});
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
