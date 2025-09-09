import express from "express";
import pg from "pg";
import bcrypt from "bcrypt";
import cors from "cors";



const app = express();
app.use(cors());
app.use(express.json());


const pool = new pg.Pool({
    user: 'postgres', 
    host: 'localhost',
    database: 'jaff', 
    password: 'root', 
    port: 5432,
});

// Register user
app.post("/api/register", async (req, res) => {
  const { username, password } = req.body;

  try {
    const result = await pool.query(
      "INSERT INTO user_login (username, password) VALUES ($1, $2) RETURNING id, username",
      [username, password]
    );
    res.json({ success: true, user: result.rows[0] });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: "User registration failed" });
  }
});

// Login user
app.post("/login", async (req, res) => {
  const { username, password } = req.body;
  console.log(req.body);

  try {
    const result = await pool.query(
      "SELECT * FROM user_login WHERE username = $1",
      [username]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({ success: false, error: "User not found" });
    }

    const user = result.rows[0];

    // Direct string comparison (no bcrypt)
    if (user.password !== password) {
      return res.status(401).json({ success: false, error: "Invalid password" });
    }

    res.json({
      success: true,
      user: { id: user.id, username: user.username },
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: "Login failed" });
  }
});

// Get user's chat history by username
app.get("/api/chat-history/:username", async (req, res) => {
  const { username } = req.params;

  try {
    const result = await pool.query(
      "SELECT * FROM chat_history WHERE username = $1 ORDER BY timestamp DESC",
      [username]
    );
    res.json({ success: true, history: result.rows });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: "Failed to fetch chat history" });
  }
});

// Save chat message
app.post("/api/save-chat", async (req, res) => {
  const { username, message, sender, timestamp } = req.body;

  try {
    const result = await pool.query(
      "INSERT INTO chat_history (username, message, sender) VALUES ($1, $2, $3) RETURNING *",
      [username, message, sender]
    );
    res.json({ success: true, chat: result.rows[0] });
  } catch (error) {
    console.error(error);
    res.status(500).json({ success: false, error: "Failed to save chat" });
  }
});

app.listen(5000, () => {
  console.log("Server running on http://localhost:5000");
});