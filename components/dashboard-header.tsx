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
  const [sliderStyle, setSliderStyle] = useState({ left: 0, width: 0, opacity: 0 })
  const [isTransitioning, setIsTransitioning] = useState(false)
  const prevPathname = useRef(pathname)
  const navRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const updateSliderPosition = () => {
      const activeIndex = navItems.findIndex((item) => item.href === pathname)
      if (activeIndex !== -1 && navRef.current) {
        const navElement = document.getElementById(`nav-item-${activeIndex}`)
        if (navElement) {
          const newStyle = {
            left: navElement.offsetLeft,
            width: navElement.offsetWidth,
            opacity: 1
          }

          // If this is a route change, add transition
          if (prevPathname.current !== pathname) {
            setIsTransitioning(true)
            setSliderStyle(newStyle)
            
            // Reset transition state after animation completes
            setTimeout(() => setIsTransitioning(false), 500)
          } else {
            // Initial load - no transition
            setSliderStyle(newStyle)
          }
          
          prevPathname.current = pathname
        }
      }
    }

    // Initial position
    updateSliderPosition()

    // Update on window resize
    window.addEventListener('resize', updateSliderPosition)
    return () => window.removeEventListener('resize', updateSliderPosition)
  }, [pathname])

  return (
    <header className="fixed top-4 left-1/2 z-50 w-[95%] max-w-4xl -translate-x-1/2 sm:top-6">
      <div 
        ref={navRef}
        className="glass-effect relative flex h-14 sm:h-16 items-center justify-between gap-1 sm:gap-2 rounded-2xl border border-glass-border px-2 sm:px-8 shadow-2xl backdrop-blur-xl"
      >
        {/* Glass Slider with Zoom Effect */}
        <div
          className={`absolute h-10 sm:h-12 rounded-xl bg-gradient-to-r from-primary/40 to-accent/40 backdrop-blur-lg border border-white/20 shadow-lg ${
            isTransitioning ? 'transition-all duration-500 ease-out' : ''
          }`}
          style={{
            left: `${sliderStyle.left}px`,
            width: `${sliderStyle.width}px`,
            top: "50%",
            transform: "translateY(-50%) scale(1.02)",
            opacity: sliderStyle.opacity,
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
              className={`relative z-10 flex flex-col items-center gap-0.5 sm:gap-1 rounded-xl px-3 sm:px-6 py-2 transition-all duration-300 group ${
                isActive 
                  ? "text-foreground transform scale-105" 
                  : "text-muted-foreground hover:text-foreground hover:scale-102"
              }`}
            >
              {/* Icon Container with Enhanced Glass Effect */}
              <div className={`
                relative p-2 rounded-lg transition-all duration-300
                ${isActive 
                  ? 'bg-white/20 backdrop-blur-sm border border-white/30 shadow-lg' 
                  : 'bg-transparent group-hover:bg-white/10 group-hover:backdrop-blur-sm'
                }
              `}>
                <Icon className={`h-4 w-4 sm:h-5 sm:w-5 transition-all duration-300 ${
                  isActive ? "scale-110" : "group-hover:scale-105"
                }`} />
                
                {/* Subtle glow effect for active state */}
                {isActive && (
                  <div className="absolute inset-0 rounded-lg bg-gradient-to-r from-primary/20 to-accent/20 blur-sm -z-10" />
                )}
              </div>
              
              {/* Text with better mobile responsiveness */}
              <span className="text-[10px] sm:text-xs font-medium whitespace-nowrap transition-all duration-300">
                {item.name}
              </span>

              {/* Enhanced Hover Effect */}
              <div className={`
                absolute inset-0 rounded-xl bg-gradient-to-r from-primary/5 to-accent/5 opacity-0 transition-opacity duration-300 -z-10
                ${isActive ? 'opacity-100' : 'group-hover:opacity-100'}
              `} />
            </Link>
          )
        })}
      </div>
    </header>
  )
}