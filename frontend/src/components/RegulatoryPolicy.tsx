import React from 'react';
import Card from './Card';

interface Policy {
  title: string;
  source?: string;
  country?: string;
  domain?: string;
  regulator?: string;
}

interface RegulatoryPolicyProps {
  policies?: Policy[];
  uploadedFile: File | null;
}

const RegulatoryPolicy: React.FC<RegulatoryPolicyProps> = ({ policies = [], uploadedFile }) => {
  // Get top 5 policies by title
  const topPolicies = policies.slice(0, 5);

  if (!uploadedFile) {
    return (
      <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2">
        <div className="flex flex-col h-full gap-4">
          <h3 className="text-lg text-[#1975d4] font-bold">Top 5 Relevant Policies</h3>
          
          <div className="space-y-2 flex-1">
            {/* Placeholder policy items */}
            {Array.from({ length: 5 }).map((_, index) => (
              <div 
                key={index} 
                className="bg-gray-50 rounded-lg p-2 border border-dashed border-gray-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-gray-400 flex-shrink-0">
                        #{index + 1}
                      </span>
                      <p className="text-xs text-gray-400 italic">
                        Upload document to see policies
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <img
          src="/icons/info.svg"
          alt="Info"
          className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
        />
      </Card>
    );
  }

  if (topPolicies.length === 0) {
    return (
      <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2">
        <div className="flex flex-col h-full gap-4">
          <h3 className="text-lg text-[#1975d4] font-bold">Top 5 Relevant Policies</h3>
          
          <div className="space-y-2 flex-1">
            {/* Empty state policy items */}
            {Array.from({ length: 5 }).map((_, index) => (
              <div 
                key={index} 
                className="bg-gray-50 rounded-lg p-2 border border-dashed border-gray-300"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-xs font-medium text-gray-400 flex-shrink-0">
                        #{index + 1}
                      </span>
                      <p className="text-xs text-gray-400 italic">
                        No policies found
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
        <img
          src="/icons/info.svg"
          alt="Info"
          className="absolute top-2 right-2 w-4 h-4 cursor-pointer"
        />
      </Card>
    );
  }

  return (
    <Card className="bg-white rounded-2xl shadow-lg p-5 h-full custom-border p-2 h-full">
      <div className="flex flex-col h-full gap-4">
        <h3 className="text-lg text-[#1975d4] font-bold">Top 5 Relevant Policies</h3>
        
        <div className="space-y-2">
          {topPolicies.map((policy, index) => (
            <div 
              key={index} 
              className="bg-white rounded-lg p-2 hover:shadow-md transition-shadow"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-xs font-medium text-gray-600 flex-shrink-0">
                      #{index + 1}
                    </span>
                    <p className="text-xs font-semibold text-gray-800 leading-tight truncate" title={policy.title}>
                      {policy.title}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {policies.length > 5 && (
          <div className="mt-4 text-center">
            <p className="text-xs text-gray-500">
              Showing top 5 of {policies.length} relevant policies
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default RegulatoryPolicy;
