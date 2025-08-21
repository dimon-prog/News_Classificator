document.getElementById("checkBtn").addEventListener("click", async () => {
    const sentence = document.getElementById("inputSentence").value;
    try {
        const response = await fetch("http://localhost:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sentence })
        });
        const data = await response.json();
        document.getElementById("result").innerText = data.prediction;
    } catch (err) {
        document.getElementById("result").innerText = "Ошибка: " + err;
    }
});