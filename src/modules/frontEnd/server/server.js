import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import chat from "./chat.js";

dotenv.config();

const app = express();
app.use(cors());

const PORT = 5005;

app.get("/chat", async (req, res) => {
    const resp = await chat(req.query.question);
    res.send(resp);
});

app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});
