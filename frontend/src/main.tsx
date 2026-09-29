import React, { useEffect } from "react"
import ReactDOM from "react-dom/client"
import App from "./App"
import "./index.css"
import "./app-motion.css"
import { AuthProvider } from "@/lib/auth"

declare global {
  interface Window { __falconLoader?: { appReady: () => void } }
}

// Сообщает загрузочному экрану (index.html), что приложение смонтировано.
function LoaderReady() {
  useEffect(() => { window.__falconLoader?.appReady() }, [])
  return null
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AuthProvider><App /><LoaderReady /></AuthProvider>
  </React.StrictMode>,
)
