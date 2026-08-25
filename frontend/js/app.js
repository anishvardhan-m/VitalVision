const socket = io();

const backendStatus = document.getElementById("backend-status");

socket.on("connect", () => {
    console.log("Connected to VitalVision backend.");

    backendStatus.textContent = "Backend connected";
});

socket.on("backend_status", (data) => {
    console.log("Backend status:", data);

    backendStatus.textContent = data.message;
});

socket.on("disconnect", () => {
    console.log("Disconnected from VitalVision backend.");

    backendStatus.textContent = "Backend disconnected";
});