import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import GalleryPage from "@/components/gallery-page";
import SearchWorkspace from "@/components/search-workspace-live";
import AuthPage from "@/components/auth-page";
import AnalyticsPage from "@/components/analytics-page";
import { ProtectedRoute } from "@/components/protected-route";
import {
  ApiSection,
  FinalCta,
  HeroSection,
  LandingFooter,
  LandingHeader,
  LandingMotion,
  TechnologySection,
  TrustStrip,
  UseCasesSection,
} from "@/components/landing";

export default function App() {
  if (window.location.pathname.startsWith("/auth")) {
    return <TooltipProvider delayDuration={150}><AuthPage /><Toaster position="bottom-right" /></TooltipProvider>;
  }

  if (window.location.pathname.startsWith("/gallery")) {
    return (
      <TooltipProvider delayDuration={150}>
        <ProtectedRoute><GalleryPage /></ProtectedRoute>
        <Toaster position="bottom-right" />
      </TooltipProvider>
    );
  }

  if (window.location.pathname.startsWith("/analytics")) {
    return (
      <TooltipProvider delayDuration={150}>
        <ProtectedRoute><AnalyticsPage /></ProtectedRoute>
        <Toaster position="bottom-right" />
      </TooltipProvider>
    );
  }

  if (window.location.pathname.startsWith("/app")) {
    return (
      <TooltipProvider delayDuration={150}>
        <ProtectedRoute><SearchWorkspace /></ProtectedRoute>
        <Toaster position="bottom-right" />
      </TooltipProvider>
    );
  }

  return (
    <TooltipProvider delayDuration={150}>
      <a href="#main" className="skip-link">
        Перейти к содержимому
      </a>
      <LandingMotion />
      <LandingHeader />
      <main id="main" className="page-grid">
        <div className="hero-continuum">
          <HeroSection />
          <UseCasesSection />
        </div>
        <TechnologySection />
        <TrustStrip />
        <ApiSection />
        <FinalCta />
      </main>
      <LandingFooter />
      <Toaster position="bottom-right" />
    </TooltipProvider>
  );
}
