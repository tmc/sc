/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    darkMode: 'class', // or 'media' if preferred, but 'class' gives manual control
    theme: {
        extend: {
            fontFamily: {
                sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
                mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'monospace'],
            },
            colors: {
                // Midnight Neon Palette
                midnight: {
                    950: '#020617', // Deepest background
                    925: '#050a1f', // Slightly lighter panel bg
                    900: '#0f172a',
                    800: '#1e293b',
                    700: '#334155',
                },
                neon: {
                    blue: '#00f0ff',
                    purple: '#b026ff',
                    pink: '#ff00aa',
                    green: '#39ff14',
                },
                glass: {
                    light: 'rgba(255, 255, 255, 0.7)',
                    dark: 'rgba(15, 23, 42, 0.6)',
                    border: 'rgba(255, 255, 255, 0.08)',
                    highlight: 'rgba(255, 255, 255, 0.05)',
                }
            },
            backdropBlur: {
                xs: '2px',
                xl: '24px',
            },
            boxShadow: {
                'neon': '0 0 10px rgba(0, 240, 255, 0.5), 0 0 20px rgba(0, 240, 255, 0.3)',
                'neon-strong': '0 0 15px rgba(0, 240, 255, 0.6), 0 0 30px rgba(0, 240, 255, 0.4)',
                'neon-purple': '0 0 10px rgba(176, 38, 255, 0.5), 0 0 20px rgba(176, 38, 255, 0.3)',
                'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.3)',
                'glass-inset': 'inset 0 0 20px rgba(255, 255, 255, 0.02)',
                'glass-sm': '0 4px 16px 0 rgba(0, 0, 0, 0.2)',
            },
            animation: {
                'fade-in': 'fadeIn 0.3s ease-out',
                'slide-up': 'slideUp 0.4s ease-out',
                'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
                'glow-pulse': 'glowPulse 2s ease-in-out infinite',
            },
            keyframes: {
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' },
                },
                slideUp: {
                    '0%': { transform: 'translateY(10px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' },
                },
                glowPulse: {
                    '0%, 100%': { boxShadow: '0 0 5px rgba(0, 240, 255, 0.2)' },
                    '50%': { boxShadow: '0 0 20px rgba(0, 240, 255, 0.6)' },
                }
            }
        },
    },
    plugins: [],
}
