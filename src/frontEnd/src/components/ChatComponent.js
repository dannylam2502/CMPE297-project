import React, { useState } from "react";
import axios from "axios";
import { Input } from "antd";

const { Search } = Input;

const DOMAIN = "http://localhost:5005";

const searchContainer = {
    display: "flex",
    justifyContent: "center",
};

const ChatComponent = (props) => {
    const { handleResp, addQuestion, isLoading, setIsLoading } = props;
    const [searchValue, setSearchValue] = useState("");

    const onSearch = async (question) => {
        if (!question.trim()) return; // Prevent empty submissions

        setSearchValue("");
        addQuestion(question); // Add the question to the conversation immediately
        setIsLoading(true);

        try {
            const response = await axios.get(`${DOMAIN}/chat`, {
                params: {
                    question,
                },
            });
            handleResp(question, response.data);
        } catch (error) {
            console.error(`Error: ${error}`);
            handleResp(question, "An error occurred while fetching the answer.");
        } finally {
            setIsLoading(false);
        }
    };

    const handleChange = (e) => {
        setSearchValue(e.target.value);
    };

    return (
        <div style={searchContainer}>
            <Search
                placeholder="Ask your question..."
                enterButton="Send"
                size="large"
                onSearch={onSearch}
                loading={isLoading}
                value={searchValue}
                onChange={handleChange}
            />
        </div>
    );
};

export default ChatComponent;
