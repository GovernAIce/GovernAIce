"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  Upload,
  Search,
  X,
  ChevronDown,
  Info,
  ChevronLeft,
  ChevronRight,
  Send,
  Home,
  FileText,
  Target,
  Shield,
  BarChart3,
  Users,
  Building,
  Folder,
  User,
} from "lucide-react"

export default function GovernAIceDashboard() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedCountry, setSelectedCountry] = useState("Country")
  const [selectedDomain, setSelectedDomain] = useState("Domain")

  return (
    <div className="flex h-screen bg-[#f5f5f5]">
      {/* Left Sidebar */}
      <div className="w-64 bg-[#ffffff] border-r border-[#d9d9d9] flex flex-col">
        {/* Logo */}
        <div className="p-6 border-b border-[#f0f0f0]">
          <h1 className="text-[#537ff1] text-xl font-semibold">GovernAIce</h1>
        </div>

        {/* Navigation */}
        <div className="flex-1 overflow-y-auto">
          {/* Menu Section */}
          <div className="p-4">
            <h3 className="text-[#9ea2ae] text-sm font-medium mb-3">Menu</h3>
            <div className="space-y-1">
              <Button variant="ghost" className="w-full justify-start gap-3 bg-[#f0f0f0] text-[#000000]">
                <Home className="w-4 h-4" />
                Home
              </Button>
            </div>
          </div>

          {/* Tools Section */}
          <div className="p-4">
            <h3 className="text-[#9ea2ae] text-sm font-medium mb-3">Tools</h3>
            <div className="space-y-1">
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <FileText className="w-4 h-4" />
                Policy Analysis
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <Target className="w-4 h-4" />
                Opportunity Identification
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <Shield className="w-4 h-4" />
                Compliance Risk Assessment
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <BarChart3 className="w-4 h-4" />
                Reports Generation
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <Users className="w-4 h-4" />
                Policy Engagement
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <Building className="w-4 h-4" />
                Vendor Management
              </Button>
            </div>
          </div>

          {/* Project Folders */}
          <div className="p-4">
            <h3 className="text-[#9ea2ae] text-sm font-medium mb-3">Project Folders:</h3>
            <div className="space-y-1">
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <Folder className="w-4 h-4" />
                GovernAIce
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <Folder className="w-4 h-4" />
                UX@Berkeley
              </Button>
            </div>
          </div>

          {/* Reports */}
          <div className="p-4">
            <h3 className="text-[#9ea2ae] text-sm font-medium mb-3">Reports:</h3>
            <div className="space-y-1">
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#537ff1] hover:text-[#537ff1]">
                <FileText className="w-4 h-4" />
                Feb. 28th- 9:46am
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <FileText className="w-4 h-4" />
                Jan. 5th- 11:53am
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <FileText className="w-4 h-4" />
                Jan. 7th- 12:11am
              </Button>
              <Button variant="ghost" className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]">
                <FileText className="w-4 h-4" />
                Jan. 30th- 6:40pm
              </Button>
            </div>
          </div>

          {/* Team Members */}
          <div className="p-4">
            <h3 className="text-[#9ea2ae] text-sm font-medium mb-3">Team Members</h3>
            <div className="space-y-1">
              {Array.from({ length: 5 }).map((_, i) => (
                <Button
                  key={i}
                  variant="ghost"
                  className="w-full justify-start gap-3 text-[#9ea2ae] hover:text-[#000000]"
                >
                  <User className="w-4 h-4" />
                  Name Name
                </Button>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom User */}
        <div className="p-4 border-t border-[#f0f0f0]">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-[#d9d9d9] rounded-full flex items-center justify-center">
              <User className="w-5 h-5 text-[#9ea2ae]" />
            </div>
            <span className="text-[#000000] font-medium">Name Name</span>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="bg-[#ffffff] border-b border-[#d9d9d9] p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-8">
              <h2 className="text-[#537ff1] text-xl font-medium">Good Evening, Yan</h2>
              <h3 className="text-[#537ff1] text-xl font-medium">Project Name</h3>
            </div>
            <div className="flex items-center gap-4">
              <span className="text-[#9ea2ae] text-sm">last edit: Feb. 18th- 9:46am</span>
              <div className="flex items-center gap-2">
                <span className="text-[#000000] font-medium">Edit Mode</span>
                <div className="w-12 h-6 bg-[#537ff1] rounded-full flex items-center justify-end p-1">
                  <div className="w-4 h-4 bg-[#ffffff] rounded-full"></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Main Dashboard Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-2 gap-6 mb-6">
            {/* Upload Project */}
            <Card className="border-[#d9d9d9]">
              <CardContent className="p-6">
                <div className="flex items-start gap-4">
                  <div className="w-16 h-16 bg-[#3cc3df] rounded-xl flex items-center justify-center">
                    <Upload className="w-8 h-8 text-[#ffffff]" />
                  </div>
                  <div>
                    <h3 className="text-[#537ff1] text-lg font-semibold mb-2">Upload Project</h3>
                    <p className="text-[#9ea2ae] text-sm">Please upload your product illustration: doc, pdf, jpg...</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Explore Policy */}
            <Card className="border-[#d9d9d9]">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[#537ff1] text-lg font-semibold">Explore Policy</h3>
                  <Info className="w-4 h-4 text-[#537ff1]" />
                </div>
                <div className="space-y-3">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-[#9ea2ae]" />
                    <Input
                      placeholder="Search"
                      className="pl-10 pr-10 border-[#d9d9d9]"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    {searchQuery && (
                      <X
                        className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-[#9ea2ae] cursor-pointer"
                        onClick={() => setSearchQuery("")}
                      />
                    )}
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      className="border-[#d9d9d9] text-[#9ea2ae] hover:text-[#000000] bg-transparent"
                    >
                      {selectedCountry} <ChevronDown className="w-4 h-4 ml-1" />
                    </Button>
                    <Button
                      variant="outline"
                      className="border-[#d9d9d9] text-[#9ea2ae] hover:text-[#000000] bg-transparent"
                    >
                      {selectedDomain} <ChevronDown className="w-4 h-4 ml-1" />
                    </Button>
                  </div>
                  <Button className="w-full bg-[#537ff1] hover:bg-[#1769db] text-[#ffffff]">Compare</Button>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-2 gap-6 mb-6">
            {/* Overall Score */}
            <Card className="border-[#d9d9d9]">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[#537ff1] text-lg font-semibold">Overall Score: 85</h3>
                  <Info className="w-4 h-4 text-[#537ff1]" />
                </div>
                <p className="text-[#9ea2ae] text-sm mb-6">
                  Compare use case to the target policies and give an overall score about the compliance level
                </p>
                <div className="flex justify-center">
                  <div className="w-64 h-64 relative">
                    <svg viewBox="0 0 200 200" className="w-full h-full">
                      {/* Radar chart background */}
                      <polygon
                        points="100,20 150,50 170,100 150,150 100,180 50,150 30,100 50,50"
                        fill="none"
                        stroke="#d9d9d9"
                        strokeWidth="1"
                      />
                      <polygon
                        points="100,40 130,60 150,100 130,140 100,160 70,140 50,100 70,60"
                        fill="none"
                        stroke="#d9d9d9"
                        strokeWidth="1"
                      />
                      <polygon
                        points="100,60 110,80 130,100 110,120 100,140 90,120 70,100 90,80"
                        fill="none"
                        stroke="#d9d9d9"
                        strokeWidth="1"
                      />

                      {/* Data polygon */}
                      <polygon
                        points="100,30 140,65 160,100 140,135 100,170 60,135 40,100 60,65"
                        fill="#537ff1"
                        fillOpacity="0.3"
                        stroke="#537ff1"
                        strokeWidth="2"
                      />

                      {/* Axis lines */}
                      <line x1="100" y1="100" x2="100" y2="20" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="150" y2="50" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="170" y2="100" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="150" y2="150" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="100" y2="180" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="50" y2="150" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="30" y2="100" stroke="#d9d9d9" strokeWidth="1" />
                      <line x1="100" y1="100" x2="50" y2="50" stroke="#d9d9d9" strokeWidth="1" />
                    </svg>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Relevant Policies & Regulators */}
            <Card className="border-[#d9d9d9]">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[#537ff1] text-lg font-semibold">Relevant Policies & Regulators</h3>
                  <Info className="w-4 h-4 text-[#537ff1]" />
                </div>
                <p className="text-[#9ea2ae] text-sm mb-6">(pop up relevant information)</p>

                {/* Bar Chart */}
                <div className="h-48 flex items-end justify-center gap-2 mb-4">
                  {/* 2020 */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="flex gap-1">
                      <div className="w-4 h-8 bg-[#ff928a] rounded-t"></div>
                      <div className="w-4 h-6 bg-[#ffae4c] rounded-t"></div>
                      <div className="w-4 h-10 bg-[#6fd195] rounded-t"></div>
                    </div>
                    <span className="text-xs text-[#9ea2ae]">2020</span>
                  </div>

                  {/* 2021 */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="flex gap-1">
                      <div className="w-4 h-16 bg-[#ff928a] rounded-t"></div>
                      <div className="w-4 h-8 bg-[#ffae4c] rounded-t"></div>
                      <div className="w-4 h-12 bg-[#6fd195] rounded-t"></div>
                    </div>
                    <span className="text-xs text-[#9ea2ae]">2021</span>
                  </div>

                  {/* 2022 */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="flex gap-1">
                      <div className="w-4 h-20 bg-[#537ff1] rounded-t"></div>
                    </div>
                    <span className="text-xs text-[#9ea2ae]">2022</span>
                  </div>

                  {/* 2023 */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="flex gap-1">
                      <div className="w-4 h-14 bg-[#537ff1] rounded-t"></div>
                      <div className="w-4 h-16 bg-[#3cc3df] rounded-t"></div>
                    </div>
                    <span className="text-xs text-[#9ea2ae]">2023</span>
                  </div>

                  {/* 2024 */}
                  <div className="flex flex-col items-center gap-1">
                    <div className="flex gap-1">
                      <div className="w-4 h-18 bg-[#537ff1] rounded-t"></div>
                      <div className="w-4 h-10 bg-[#6fd195] rounded-t"></div>
                    </div>
                    <span className="text-xs text-[#9ea2ae]">2024</span>
                  </div>
                </div>

                {/* Legend */}
                <div className="flex justify-center gap-4 text-xs">
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-[#ff928a] rounded"></div>
                    <span className="text-[#9ea2ae]">Figma</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <div className="w-3 h-3 bg-[#537ff1] rounded"></div>
                    <span className="text-[#9ea2ae]">AI</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div className="grid grid-cols-2 gap-6">
            {/* Excellencies */}
            <Card className="border-[#d9d9d9]">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[#537ff1] text-lg font-semibold">Excellencies</h3>
                  <Info className="w-4 h-4 text-[#537ff1]" />
                </div>
                <p className="text-[#9ea2ae] text-sm">
                  Identify and list which parts are well-written in the use case, compared to the selected policies,
                  such as measures to risks.
                </p>
              </CardContent>
            </Card>

            {/* Major Gaps */}
            <Card className="border-[#d9d9d9]">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-[#537ff1] text-lg font-semibold">Major Gaps</h3>
                  <Info className="w-4 h-4 text-[#537ff1]" />
                </div>
                <p className="text-[#9ea2ae] text-sm">
                  Identify and list what are missed in the use case, compared to the selected policies.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      {/* Right Sidebar */}
      <div className="w-80 bg-[#ffffff] border-l border-[#d9d9d9] flex flex-col">
        {/* NEW Badge */}
        <div className="absolute top-4 right-4 z-10">
          <Badge className="bg-[#537ff1] text-[#ffffff] rounded-full px-3 py-1">NEW</Badge>
        </div>

        {/* Relevant Policies Section */}
        <div className="p-6 border-b border-[#f0f0f0]">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-[#537ff1] text-lg font-semibold">Relevant Policies</h3>
            <X className="w-4 h-4 text-[#9ea2ae] cursor-pointer" />
          </div>
          <p className="text-[#000000] text-sm leading-relaxed mb-4">
            Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et
            dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex
            ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat
            nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt... ut
            aliquip ex ea commodo.
          </p>
          <div className="flex items-center justify-between mb-4">
            <ChevronLeft className="w-5 h-5 text-[#9ea2ae] cursor-pointer" />
            <Button className="bg-[#537ff1] hover:bg-[#1769db] text-[#ffffff] text-sm px-4 py-2">
              View Full Policy Document
            </Button>
            <ChevronRight className="w-5 h-5 text-[#9ea2ae] cursor-pointer" />
          </div>
        </div>

        {/* Chat Section */}
        <div className="flex-1 flex flex-col">
          <div className="p-6 border-b border-[#f0f0f0]">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 bg-[#537ff1] rounded-full flex items-center justify-center">
                <span className="text-[#ffffff] text-sm">😊</span>
              </div>
              <div>
                <h3 className="text-[#537ff1] text-lg font-semibold">Chat with Me</h3>
                <Info className="w-4 h-4 text-[#537ff1] inline" />
              </div>
            </div>
            <p className="text-[#000000] text-sm">I am Govii, how can I help you with compliance project today?</p>
          </div>

          {/* Chat Input */}
          <div className="mt-auto p-6">
            <div className="relative">
              <Input placeholder="Type Here" className="pr-10 border-[#d9d9d9] rounded-full" />
              <Send className="absolute right-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-[#537ff1] cursor-pointer" />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
