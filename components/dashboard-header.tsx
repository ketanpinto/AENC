"use client"

import { Home, BarChart3, FileText, Truck } from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useEffect, useState, useRef } from "react"

const navItems = [
  { name: "Home", href: "/", icon: Home },
  { name: "Predict & Analyze", href: "/predict-analyze", icon: BarChart3 },
  { name: "Contract Analysis", href: "/contract-analysis", icon: FileText },
  { name: "Supply Chain", href: "/supply-chain", icon: Truck },
]

export function DashboardHeader() {
  const pathname = usePathname()
  const [sliderStyle, setSliderStyle] = useState({ left: 0, width: 0 })
  const isInitialMount = useRef(true)

  useEffect(() => {
    const activeIndex = navItems.findIndex((item) => item.href === pathname)
    if (activeIndex !== -1) {
      const navElement = document.getElementById(`nav-item-${activeIndex}`)
      if (navElement) {
        if (isInitialMount.current) {
          setSliderStyle({
            left: navElement.offsetLeft,
            width: navElement.offsetWidth,
          })
          isInitialMount.current = false
        } else {
          setSliderStyle({
            left: navElement.offsetLeft,
            width: navElement.offsetWidth,
          })
        }
      }
    }
  }, [pathname])

  return (
    <header className="fixed top-6 left-1/2 z-50 w-[95%] max-w-4xl -translate-x-1/2">
      <div className="glass-effect relative flex h-16 items-center justify-center gap-2 rounded-full border border-glass-border px-8 shadow-2xl backdrop-blur-xl">
        <div
          className={`absolute h-12 rounded-full bg-gradient-to-r from-primary/30 to-accent/30 backdrop-blur-sm ${
            isInitialMount.current ? "" : "transition-all duration-500 ease-in-out"
          }`}
          style={{
            left: `${sliderStyle.left}px`,
            width: `${sliderStyle.width}px`,
            top: "50%",
            transform: "translateY(-50%)",
          }}
        />

        {navItems.map((item, index) => {
          const Icon = item.icon
          const isActive = pathname === item.href
          return (
            <Link
              key={item.href}
              href={item.href}
              id={`nav-item-${index}`}
              className={`relative z-10 flex flex-col items-center gap-1 rounded-full px-6 py-2 transition-all duration-300 ${
                isActive ? "text-foreground" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon className={`h-5 w-5 ${isActive ? "scale-110" : ""} transition-transform`} />
              <span className="text-xs font-medium">{item.name}</span>
            </Link>
          )
        })}
      </div>
    </header>
  )
}
