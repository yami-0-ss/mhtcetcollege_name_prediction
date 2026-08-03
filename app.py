from flask import Flask, request, jsonify, render_template_string
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
MODEL_PATH = "collegename_model.pkl"
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Single-file HTML template with Glassmorphism UI, Tailwind CSS, and Lucide icons
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MHTCET College Admission Predictor</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: radial-gradient(circle at 50% -20%, #1e1b4b, #0f172a, #020617);
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .glow-btn {
            transition: all 0.3s ease;
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
        }

        .glow-btn:hover {
            box-shadow: 0 0 35px rgba(99, 102, 241, 0.8);
            transform: translateY(-2px);
        }

        @keyframes pulseGlow {
            0%, 100% { opacity: 0.4; }
            50% { opacity: 0.8; }
        }

        .ambient-glow {
            animation: pulseGlow 6s infinite ease-in-out;
        }
    </style>
</head>
<body class="min-h-screen text-slate-100 flex items-center justify-center p-4 relative overflow-x-hidden">

    <div class="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-indigo-600/20 rounded-full blur-[120px] pointer-events-none ambient-glow"></div>
    <div class="absolute bottom-10 right-10 w-[300px] h-[300px] bg-violet-600/20 rounded-full blur-[100px] pointer-events-none"></div>

    <div class="w-full max-w-4xl z-10 my-8">
        <div class="text-center mb-8">
            <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-sm font-medium mb-3">
                <i data-lucide="sparkles" class="w-4 h-4"></i> AI-Powered Machine Learning Model
            </div>
            <h1 class="text-4xl md:text-5xl font-extrabold bg-clip-text text-transparent bg-gradient-to-r from-indigo-200 via-sky-300 to-indigo-400 tracking-tight">
                College Predictor Portal
            </h1>
            <p class="text-slate-400 mt-2 text-sm md:text-base">Enter your details below to predict cutoff and seat allocation results.</p>
        </div>

        <div class="glass-card rounded-2xl p-6 md:p-10 shadow-2xl">
            <form id="predictionForm" class="grid grid-cols-1 md:grid-cols-2 gap-6">
                
                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Merit Number</label>
                    <div class="relative">
                        <i data-lucide="hash" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="merit_num" required placeholder="e.g. 12450" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">MHTCET Percentile</label>
                    <div class="relative">
                        <i data-lucide="percent" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" step="0.0001" name="mhtcet_percentile" required placeholder="e.g. 98.4521" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Name Code</label>
                    <div class="relative">
                        <i data-lucide="user" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="name_code" required placeholder="Encoded integer" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Gender Code</label>
                    <div class="relative">
                        <i data-lucide="users" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="gender_code" required placeholder="0 for Male, 1 for Female" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Category Code</label>
                    <div class="relative">
                        <i data-lucide="layers" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="category_code" required placeholder="Encoded integer category" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Seat Allotted Code</label>
                    <div class="relative">
                        <i data-lucide="check-square" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="seat_alloted" required placeholder="Encoded integer" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Course Name Code</label>
                    <div class="relative">
                        <i data-lucide="book-open" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="course_code" required placeholder="Encoded integer course" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div>
                    <label class="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-2">Seat Number Code</label>
                    <div class="relative">
                        <i data-lucide="award" class="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400"></i>
                        <input type="number" name="seat_num" required placeholder="Encoded integer seat num" class="w-full bg-slate-900/60 border border-slate-700/80 rounded-xl pl-10 pr-4 py-3 text-slate-100 placeholder-slate-500 focus:outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 transition">
                    </div>
                </div>

                <div class="md:col-span-2 mt-4">
                    <button type="submit" class="glow-btn w-full bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 hover:from-indigo-600 hover:to-pink-600 text-white font-bold py-4 rounded-xl flex items-center justify-center gap-2 transition text-base">
                        <i data-lucide="cpu" class="w-5 h-5"></i> Execute Model Prediction
                    </button>
                </div>
            </form>

            <div id="loading" class="hidden text-center my-6">
                <div class="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400"></div>
                <p class="text-sm text-slate-400 mt-2">Processing input values through RandomForest model...</p>
            </div>

            <div id="resultBox" class="hidden mt-8 p-6 bg-gradient-to-br from-emerald-950/40 to-slate-900 border border-emerald-500/30 rounded-xl text-center">
                <p class="text-xs uppercase font-bold text-emerald-400 tracking-widest">Prediction Result</p>
                <div id="resultValue" class="text-2xl md:text-3xl font-extrabold text-slate-100 mt-2"></div>
            </div>
        </div>
    </div>

    <script>
        lucide.createIcons();

        document.getElementById("predictionForm").addEventListener("submit", async function (e) {
            e.preventDefault();

            const form = e.target;
            const formData = new FormData(form);
            const loading = document.getElementById("loading");
            const resultBox = document.getElementById("resultBox");
            const resultValue = document.getElementById("resultValue");

            loading.classList.remove("hidden");
            resultBox.classList.add("hidden");

            try {
                const response = await fetch("/predict", {
                    method: "POST",
                    body: formData
                });

                const data = await response.json();
                loading.classList.add("hidden");

                if (data.success) {
                    resultValue.innerText = data.prediction;
                    resultBox.classList.remove("hidden");
                } else {
                    alert("Prediction Error: " + data.error);
                }
            } catch (err) {
                loading.classList.add("hidden");
                alert("Server Connection Failed.");
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"success": False, "error": "Model file not loaded."}), 500

    try:
        # Extract features matching model expectations[cite: 1]
        merit_num = float(request.form.get("merit_num", 0))
        mhtcet_percentile = float(request.form.get("mhtcet_percentile", 0))
        name_code = float(request.form.get("name_code", 0))
        gender_code = float(request.form.get("gender_code", 0))
        category_code = float(request.form.get("category_code", 0))
        seat_alloted = float(request.form.get("seat_alloted", 0))
        course_code = float(request.form.get("course_code", 0))
        seat_num = float(request.form.get("seat_num", 0))

        features = np.array([[
            merit_num,
            mhtcet_percentile,
            name_code,
            gender_code,
            category_code,
            seat_alloted,
            course_code,
            seat_num
        ]])

        prediction = model.predict(features)[0]

        return jsonify({
            "success": True, 
            "prediction": str(prediction)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
