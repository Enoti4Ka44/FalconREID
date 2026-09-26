import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { apiFetch, AUTH_TOKEN_KEY, readApiError } from "@/lib/api";

export type User = {
  id: string;
  username: string;
  created_at: string;
};

type AuthContextValue = {
  user: User | null;
  loading: boolean;
  login: (username: string, password: string) => Promise<void>;
  register: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | null>(null);
const AUTH_USER_KEY = "falcon_current_user";

function tokenSubject(token: string) {
  try {
    const encoded = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const payload = JSON.parse(atob(encoded.padEnd(Math.ceil(encoded.length / 4) * 4, "="))) as { sub?: string; exp?: number };
    if (!payload.sub || (payload.exp && payload.exp * 1000 <= Date.now())) return null;
    return payload.sub;
  } catch {
    return null;
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    const token = localStorage.getItem(AUTH_TOKEN_KEY);
    const savedUser = localStorage.getItem(AUTH_USER_KEY);
    const subject = token ? tokenSubject(token) : null;
    if (!subject || !savedUser) {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const parsed = JSON.parse(savedUser) as User;
      if (parsed.id !== subject) throw new Error("Stored user does not match token");
      setUser(parsed);
    } catch {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
      setUser(null);
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    void refresh();
    const reset = () => {
      localStorage.removeItem(AUTH_TOKEN_KEY);
      localStorage.removeItem(AUTH_USER_KEY);
      setUser(null);
    };
    window.addEventListener("falcon:unauthorized", reset);
    return () => window.removeEventListener("falcon:unauthorized", reset);
  }, [refresh]);

  const login = useCallback(async (username: string, password: string) => {
    const response = await apiFetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, "Не удалось войти в аккаунт."));
    }
    const result = await response.json() as { access_token: string };
    const id = tokenSubject(result.access_token);
    if (!id) throw new Error("Сервер вернул некорректный токен доступа.");
    const nextUser: User = { id, username: username.trim(), created_at: new Date().toISOString() };
    localStorage.setItem(AUTH_TOKEN_KEY, result.access_token);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(nextUser));
    setUser(nextUser);
  }, []);

  const register = useCallback(async (username: string, password: string) => {
    const response = await apiFetch("/api/auth/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    if (!response.ok) {
      throw new Error(await readApiError(response, "Не удалось создать аккаунт."));
    }
    const registeredUser = await response.json() as User;
    await login(username, password);
    localStorage.setItem(AUTH_USER_KEY, JSON.stringify(registeredUser));
    setUser(registeredUser);
  }, [login]);

  const logout = useCallback(async () => {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    localStorage.removeItem(AUTH_USER_KEY);
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout, refresh }),
    [user, loading, login, register, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}

export function safeNextPath(value: string | null, fallback = "/app") {
  return value?.startsWith("/") && !value.startsWith("//") ? value : fallback;
}
