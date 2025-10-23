import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";
dotenv.config();

const llm = new ChatOpenAI({
    model: "gpt-4o",
    // model: "gpt-5",
    temperature: 1, // deterministic [0, 1] random
    openAIApiKey: process.env.OPENAI_API_KEY,
    // other params...
});

const chat = async (question) => {
    const aiMsg = await llm.invoke([
        {
            role: "system",
            content:
                "You are a helpful assistant who will engage in a conversation with me. Keep your reply short and concise.",
        },
        {
            role: "user",
            content: question,
        },
    ]);

    return aiMsg?.content;
};

export default chat;
