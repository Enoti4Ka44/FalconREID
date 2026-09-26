import { useEffect, type ReactNode } from "react";
import { Loader2 } from "lucide-react";
import { useAuth } from "@/lib/auth";

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, loading } = useAuth();

  useEffect(() => {
    if (!loading && !user) {
      const next = `${window.location.pathname}${window.location.search}`;
      window.location.replace(`/auth?next=${encodeURIComponent(next)}`);
    }
  }, [loading, user]);

  if (loading || !user) {
    return (
      <main className="grid min-h-screen place-items-center bg-[#050d16] text-slate-300">
        <div className="flex items-center gap-3 text-sm">
          <Loader2 className="h-4 w-4 animate-spin text-blue-300" />
          Проверяем сессию…
        </div>
      </main>
    );
  }

  return children;
}
