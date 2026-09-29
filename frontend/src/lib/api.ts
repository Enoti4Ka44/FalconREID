const configuredBase = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");
export const AUTH_TOKEN_KEY = "falcon_access_token";

export const apiUrl = (path: string) =>
  `${configuredBase}${path.startsWith("/") ? path : `/${path}`}`;

export async function apiFetch(path: string, init: RequestInit = {}) {
  const headers = new Headers(init.headers);
  const token = localStorage.getItem(AUTH_TOKEN_KEY);
  if (token && !headers.has("Authorization")) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  const response = await fetch(apiUrl(path), {
    ...init,
    headers,
  });

  if (response.status === 401 && !path.startsWith("/api/auth/")) {
    localStorage.removeItem(AUTH_TOKEN_KEY);
    window.dispatchEvent(new Event("falcon:unauthorized"));
  }

  return response;
}

export async function readApiError(response: Response, fallback: string) {
  const body = await response.json().catch(() => null) as
    | { detail?: string | Array<{ msg?: string }> }
    | null;
  const detail = body?.detail;

  if (typeof detail === "string") {
    const messages: Record<string, string> = {
      "Invalid credentials": "Неверное имя пользователя или пароль.",
      "Username already exists": "Пользователь с таким именем уже существует.",
      "Authentication required": "Войдите в аккаунт, чтобы продолжить.",
      "Invalid token": "Сессия истекла. Войдите в аккаунт ещё раз.",
    };
    return messages[detail] ?? detail;
  }

  if (Array.isArray(detail)) {
    const validation = detail.map((item) => item.msg).filter(Boolean).join("; ");
    if (validation) return validation;
  }

  return fallback;
}
