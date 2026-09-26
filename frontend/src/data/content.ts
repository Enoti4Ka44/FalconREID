import { Building2, Landmark, ShieldCheck, Truck, type LucideIcon } from "lucide-react"

export interface NavItem { label: string; href: string }
export interface UseCase { title: string; description: string; image: string; alt: string }
export interface TechnologyFeature { label: string; value: string }
export interface TrustCategory { label: string; icon: LucideIcon }
export interface CodeExample { id: "python" | "curl" | "javascript"; label: string; code: string }

export const navigation: NavItem[] = [
  { label: "Продукт", href: "#product" },
  { label: "Решения", href: "#solutions" },
  { label: "API", href: "#api" },
  { label: "Документация", href: "/docs" },
  { label: "Цены", href: "#cta" },
]

export const useCases: UseCase[] = [
  { title: "Безопасность городов", description: "Поиск автомобилей в потоке городских камер", image: "/case-city-v2.jpg", alt: "Городская магистраль с автомобилями" },
  { title: "Контроль территории", description: "Бизнес-центры, ТЦ, жилые комплексы", image: "/case-gate-v2.jpg", alt: "Автомобиль у шлагбаума" },
  { title: "Логистика и транспорт", description: "Отслеживание транспорта и автопарков", image: "/case-logistics-v2.jpg", alt: "Грузовой и легковой транспорт" },
  { title: "Частные компании", description: "Безопасность сотрудников и имущества", image: "/case-parking-v2.jpg", alt: "Защищённая парковка" },
]

export const technologyFeatures: TechnologyFeature[] = [
  { label: "Визуальный признак", value: "1536 dim" },
  { label: "Поиск", value: "Cosine ANN" },
  { label: "Решение", value: "Open-set" },
]

export const trustCategories: TrustCategory[] = [
  { label: "Правительство Москвы", icon: Landmark },
  { label: "Ситуационный центр", icon: ShieldCheck },
  { label: "Транспортные операторы", icon: Truck },
  { label: "Частные компании", icon: Building2 },
]

export const codeExamples: CodeExample[] = [
  {
    id: "python", label: "Python", code: `import requests

API_URL = "http://localhost:8000"
API_KEY = "falcon_your_api_key"

with open("car.jpg", "rb") as image:
    response = requests.post(
        f"{API_URL}/api/search",
        headers={"X-API-Key": API_KEY},
        files={"files": ("car.jpg", image, "image/jpeg")},
        data={"top_k": 10, "threshold": 0.302},
        timeout=180,
    )

response.raise_for_status()
result = response.json()
print(result["refused"], result["candidates"])`,
  },
  {
    id: "curl", label: "cURL", code: `curl --request POST \\
  "http://localhost:8000/api/search" \\
  --header "X-API-Key: falcon_your_api_key" \\
  --header "Accept: application/json" \\
  --form "files=@car.jpg;type=image/jpeg" \\
  --form "top_k=10" \\
  --form "threshold=0.302"`,
  },
  {
    id: "javascript", label: "JavaScript", code: `const form = new FormData()
form.append("files", imageFile, "car.jpg")
form.append("top_k", "10")
form.append("threshold", "0.302")

const response = await fetch(
  "http://localhost:8000/api/search",
  {
    method: "POST",
    headers: { "X-API-Key": "falcon_your_api_key" },
    body: form,
  },
)

if (!response.ok) throw new Error(await response.text())
const { refused, candidates, inference_ms } = await response.json()
console.log({ refused, candidates, inference_ms })`,
  },
]
