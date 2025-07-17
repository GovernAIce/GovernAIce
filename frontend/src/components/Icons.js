const icons = {
  Upload: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>,
  Search: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>,
  X: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M18 6 6 18M6 6l12 12"/></svg>,
  Info: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4m0-4h.01"/></svg>,
  ChevronDown: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m19 9-7 7-7-7"/></svg>,
  ChevronLeft: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m15 18-6-6 6-6"/></svg>,
  ChevronRight: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m9 18 6-6-6-6"/></svg>,
  Send: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m3 3 18 8.5L3 20V3Z"/></svg>,
  Home: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V9Z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>,
  FileText: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8l-6-6Z"/><polyline points="14 2 14 8 20 8"/><line x1="16" x2="8" y1="13" y2="13"/><line x1="16" x2="8" y1="17" y2="17"/><line x1="10" x2="8" y1="9" y2="9"/></svg>,
  Target: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>,
  Shield: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/></svg>,
  BarChart3: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="6" height="12"/><rect x="9" y="3" width="6" height="18"/><rect x="15" y="3" width="6" height="6"/></svg>,
  Users: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/></svg>,
  Building: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" x2="9" y1="3" y2="21"/><line x1="15" x2="15" y1="3" y2="21"/><line x1="3" x2="21" y1="9" y2="9"/><line x1="3" x2="21" y1="15" y2="15"/></svg>,
  Folder: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-6l-2-2H5a2 2 0 0 0-2 2Z"/></svg>,
  User: ({ className }) => <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
};