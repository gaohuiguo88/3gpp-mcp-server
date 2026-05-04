/**
 * 3GPP MCP Server - TSpec-LLM API Client
 *
 * Copyright (c) 2024, 3GPP MCP Contributors
 * Licensed under the BSD-3-Clause License
 *
 * Integrates with TSpec-LLM research: https://arxiv.org/abs/2406.01768
 */

import axios, { AxiosInstance } from 'axios';
import NodeCache from 'node-cache';
import fs from 'fs';
import path from 'path';
import MiniSearch from 'minisearch';

export interface TSpecSearchRequest {
  query: string;
  max_results?: number;
  series_filter?: string[];
  release_filter?: string[];
  specification_types?: string[];
}

export interface TSpecSearchResult {
  content: string;
  source_specification: string;
  release: string;
  section: string;
  relevance_score: number;
  metadata: {
    specification_id: string;
    working_group: string;
    document_type: string;
    keywords: string[];
  };
}

export interface TSpecSearchResponse {
  results: TSpecSearchResult[];
  total_found: number;
  query_processed: string;
  search_time_ms: number;
}

export class TSpecLLMClient {
  private api: AxiosInstance;
  private cache: NodeCache;
  private baseUrl: string;
  private dataDir: string;
  private searchIndex: MiniSearch | null = null;
  private documents: Map<string, { specId: string; release: string; content: string; title: string }> = new Map();

  constructor(dataDir?: string, huggingFaceToken?: string) {
    this.baseUrl = 'https://api-inference.huggingface.co/datasets/rasoul-nikbakht/TSpec-LLM';
    this.dataDir = dataDir || path.join(process.cwd(), 'data', 'tspec-llm');

    this.api = axios.create({
      baseURL: this.baseUrl,
      timeout: 30000,
      headers: {
        'Authorization': huggingFaceToken ? `Bearer ${huggingFaceToken}` : undefined,
        'Content-Type': 'application/json',
        'User-Agent': '3gpp-mcp-server/3.0.0'
      }
    });

    this.cache = new NodeCache({
      stdTTL: 3600, // 1 hour cache
      maxKeys: 1000,
      useClones: false
    });
  }

  /**
   * Load markdown files from the local TSpec-LLM dataset directory
   */
  private loadDocuments(): void {
    if (this.documents.size > 0) return;

    if (!fs.existsSync(this.dataDir)) {
      console.warn(`TSpec-LLM data directory not found: ${this.dataDir}`);
      return;
    }

    const files: string[] = [];

    // Recursively find all .md files
    const walkDir = (dir: string) => {
      try {
        const entries = fs.readdirSync(dir, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(dir, entry.name);
          if (entry.isDirectory()) {
            walkDir(fullPath);
          } else if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
            files.push(fullPath);
          }
        }
      } catch (err) {
        console.warn(`Cannot read directory ${dir}:`, err);
      }
    };

    walkDir(this.dataDir);

    for (const filePath of files) {
      try {
        const content = fs.readFileSync(filePath, 'utf-8');
        const relativePath = path.relative(this.dataDir, filePath);
        // Extract spec ID from filename or parent directory
        const specId = this.extractSpecId(filePath, relativePath);
        const release = this.guessRelease(content);
        const title = this.extractTitle(content);

        const docId = relativePath.replace(/\\/g, '/');
        this.documents.set(docId, { specId, release, content, title });
      } catch (err) {
        console.warn(`Cannot read ${filePath}:`, err);
      }
    }

    console.log(`Loaded ${this.documents.size} TSpec-LLM documents from ${this.dataDir}`);
  }

  /**
   * Build full-text search index (call after loadDocuments)
   */
  private ensureIndex(): MiniSearch {
    if (this.searchIndex) return this.searchIndex;

    this.loadDocuments();

    this.searchIndex = new MiniSearch({
      fields: ['content', 'title', 'specId'],
      storeFields: ['specId', 'title', 'release'],
      searchOptions: {
        boost: { title: 3, specId: 5, content: 1 },
        fuzzy: 0.2,
        prefix: true
      }
    });

    const docs = Array.from(this.documents.entries()).map(([id, doc]) => ({
      id,
      content: doc.content.slice(0, 50000), // limit per doc to avoid OOM
      title: doc.title,
      specId: doc.specId,
      release: doc.release
    }));

    if (docs.length > 0) {
      this.searchIndex.addAll(docs);
    }

    return this.searchIndex;
  }

  private extractSpecId(filePath: string, relativePath: string): string {
    // Try to extract spec ID from filename like "TS_32.290.md" or "32290.md"
    const fileName = path.basename(filePath);
    const match = fileName.match(/(?:TS[_\s]?)?(\d{2,3}[._]\d{3,4})/i);
    if (match) return `TS ${match[1].replace('_', '.')}`;
    return relativePath;
  }

  private extractTitle(content: string): string {
    const match = content.match(/^#\s+(.+)/m);
    return match ? match[1].trim() : 'Untitled';
  }

  private guessRelease(content: string): string {
    const match = content.match(/Rel(?:ease)?[\.\s-]*(\d{2,})/i);
    if (match) {
      const num = parseInt(match[1]);
      if (num >= 15) return `Rel-${num}`;
      if (num >= 8) return `Rel-${num}`;
    }
    return 'Unknown';
  }

  async searchSpecifications(request: TSpecSearchRequest): Promise<TSpecSearchResponse> {
    const cacheKey = this.generateCacheKey(request);
    const cached = this.cache.get<TSpecSearchResponse>(cacheKey);

    if (cached) {
      return cached;
    }

    try {
      const startTime = Date.now();

      // For now, simulate TSpec-LLM search using a structured approach
      // In a real implementation, this would call the Hugging Face dataset API
      const results = await this.performTSpecSearch(request);

      const response: TSpecSearchResponse = {
        results,
        total_found: results.length,
        query_processed: request.query,
        search_time_ms: Date.now() - startTime
      };

      this.cache.set(cacheKey, response);
      return response;

    } catch (error) {
      throw new Error(`TSpec-LLM search failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  private async performTSpecSearch(request: TSpecSearchRequest): Promise<TSpecSearchResult[]> {
    const query = request.query.toLowerCase();
    const maxResults = request.max_results || 10;

    // Try real local dataset search first
    const index = this.ensureIndex();
    if (this.documents.size > 0) {
      const searchResults = index.search(query, { fuzzy: 0.2, prefix: true });

      let results: TSpecSearchResult[] = searchResults.slice(0, maxResults * 2).map(result => {
        const doc = this.documents.get(result.id);
        if (!doc) return null;

        // Extract the most relevant section from content
        const section = this.findRelevantSection(doc.content, query);

        return {
          content: section,
          source_specification: doc.specId,
          release: doc.release,
          section: result.id.replace(/\\/g, '/'),
          relevance_score: result.score,
          metadata: {
            specification_id: doc.specId,
            working_group: this.guessWorkingGroup(doc.specId),
            document_type: 'Technical Specification',
            keywords: query.split(/\s+/).filter(k => k.length > 2)
          }
        } as TSpecSearchResult;
      }).filter((r): r is TSpecSearchResult => r !== null);

      // Apply filters
      if (request.series_filter && request.series_filter.length > 0) {
        results = results.filter(r =>
          request.series_filter!.some(s => r.source_specification.startsWith(`TS ${s}`))
        );
      }
      if (request.release_filter && request.release_filter.length > 0) {
        results = results.filter(r => request.release_filter!.includes(r.release));
      }

      return results.slice(0, maxResults);
    }

    // Fallback: no local dataset, send meaningful message
    console.warn('TSpec-LLM dataset not found locally. Returning empty results.');
    return [];
  }

  /**
   * Find the most relevant paragraph/section from content for the query
   */
  private findRelevantSection(content: string, query: string): string {
    const lines = content.split('\n');
    let bestScore = 0;
    let bestStart = 0;
    const queryTerms = query.toLowerCase().split(/\s+/).filter(t => t.length > 2);

    // Score each line and take the best window
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].toLowerCase();
      let score = 0;
      for (const term of queryTerms) {
        if (line.includes(term)) score++;
      }
      // Boost section headers
      if (lines[i].startsWith('#')) score *= 1.5;
      if (score > bestScore) {
        bestScore = score;
        bestStart = i;
      }
    }

    if (bestScore === 0) {
      // Return beginning of document
      return lines.slice(0, Math.min(50, lines.length)).join('\n');
    }

    // Return a window around the best matching section
    const start = Math.max(0, bestStart - 3);
    const end = Math.min(lines.length, bestStart + 30);
    return lines.slice(start, end).join('\n');
  }

  private guessWorkingGroup(specId: string): string {
    if (specId.startsWith('TS 32')) return 'SA5';
    if (specId.startsWith('TS 33')) return 'SA3';
    if (specId.startsWith('TS 38') || specId.startsWith('TS 36')) return 'RAN2';
    return 'SA2';
  }

  private generateCacheKey(request: TSpecSearchRequest): string {
    return `tspec:${JSON.stringify(request)}`;
  }

  private findFuzzySpecificationMatches(query: string): Array<{specId: string, confidence: number}> {
    const knownSpecs = [
      'TS 32.290', 'TS 32.240', 'TS 38.331', 'TS 33.501',
      'TS 23.501', 'TS 23.502', 'TS 29.594', 'TS 29.122'
    ];

    const matches: Array<{specId: string, confidence: number}> = [];

    for (const spec of knownSpecs) {
      const confidence = this.calculateSimilarity(query, spec.toLowerCase());

      // Also check for partial matches without "TS" prefix
      const specNumber = spec.replace('TS ', '');
      const partialConfidence = this.calculateSimilarity(query, specNumber.toLowerCase());

      const maxConfidence = Math.max(confidence, partialConfidence);

      if (maxConfidence > 0.5) {
        matches.push({ specId: spec, confidence: maxConfidence });
      }
    }

    // Sort by confidence and return top 3 matches
    return matches
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);
  }

  private calculateSimilarity(str1: string, str2: string): number {
    // Simple fuzzy matching using Levenshtein distance
    const longer = str1.length > str2.length ? str1 : str2;
    const shorter = str1.length > str2.length ? str2 : str1;

    if (longer.length === 0) return 1.0;

    // Check for exact substring match first
    if (longer.includes(shorter)) return 0.9;

    // Calculate Levenshtein distance
    const distance = this.levenshteinDistance(str1, str2);
    return (longer.length - distance) / longer.length;
  }

  private levenshteinDistance(str1: string, str2: string): number {
    const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));

    for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
    for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;

    for (let j = 1; j <= str2.length; j++) {
      for (let i = 1; i <= str1.length; i++) {
        const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
        matrix[j][i] = Math.min(
          matrix[j][i - 1] + 1,     // deletion
          matrix[j - 1][i] + 1,     // insertion
          matrix[j - 1][i - 1] + indicator // substitution
        );
      }
    }

    return matrix[str2.length][str1.length];
  }

  private generateSearchSuggestions(query: string): Array<{suggestion: string, reason: string}> {
    const suggestions: Array<{suggestion: string, reason: string}> = [];

    // Suggest related terms based on partial matches
    if (query.includes('notify') || query.includes('event')) {
      suggestions.push({
        suggestion: 'TS 29.594 notifyID',
        reason: 'Event exposure API with notification identifiers'
      });
      suggestions.push({
        suggestion: 'N28 interface',
        reason: 'Interface for network exposure and event notifications'
      });
    }

    if (query.includes('29') || query.includes('594')) {
      suggestions.push({
        suggestion: 'TS 29.594',
        reason: 'Network Exposure Function Northbound APIs'
      });
      suggestions.push({
        suggestion: 'event exposure',
        reason: 'Related to TS 29.594 functionality'
      });
    }

    if (query.includes('n28') || query.includes('exposure')) {
      suggestions.push({
        suggestion: 'NEF',
        reason: 'Network Exposure Function'
      });
      suggestions.push({
        suggestion: 'TS 29.122',
        reason: 'T8 reference point for Northbound APIs'
      });
    }

    if (query.includes('r15') || query.includes('rel') || query.includes('release')) {
      suggestions.push({
        suggestion: 'Release 15 specifications',
        reason: 'Search for R15-specific features'
      });
      suggestions.push({
        suggestion: 'Rel-15 baseline features',
        reason: 'Foundational 5G SA capabilities'
      });
    }

    // Add general suggestions if no specific matches
    if (suggestions.length === 0) {
      suggestions.push({
        suggestion: 'charging CHF',
        reason: 'Popular search topic - 5G charging system'
      });
      suggestions.push({
        suggestion: 'authentication 5G-AKA',
        reason: 'Popular search topic - 5G security'
      });
      suggestions.push({
        suggestion: 'handover mobility',
        reason: 'Popular search topic - radio procedures'
      });
    }

    return suggestions.slice(0, 5); // Limit to 5 suggestions
  }

  async getSpecificationInfo(specId: string): Promise<any> {
    const cacheKey = `spec_info:${specId}`;
    const cached = this.cache.get(cacheKey);

    if (cached) {
      return cached;
    }

    // Mock specification metadata
    const specInfo = {
      id: specId,
      title: this.getSpecificationTitle(specId),
      latest_release: 'Rel-17',
      working_group: this.getWorkingGroup(specId),
      status: 'Published',
      summary: this.getSpecificationSummary(specId)
    };

    this.cache.set(cacheKey, specInfo, 7200); // 2 hour cache for spec info
    return specInfo;
  }

  private getSpecificationTitle(specId: string): string {
    const titles: { [key: string]: string } = {
      'TS 32.290': '5G system; Services, operations and procedures of charging using Service Based Interface (SBI)',
      'TS 32.240': 'Telecommunication management; Charging management; Charging architecture and principles',
      'TS 38.331': '5G; NR; Radio Resource Control (RRC); Protocol specification',
      'TS 33.501': 'Security architecture and procedures for 5G System',
      'TS 23.501': 'System architecture for the 5G System (5GS)',
      'TS 23.502': 'Procedures for the 5G System (5GS)',
      'TS 29.594': '5G System; Network Exposure Function Northbound APIs; Stage 3',
      'TS 29.122': 'T8 reference point for Northbound APIs'
    };
    return titles[specId] || `${specId} - 3GPP Technical Specification`;
  }

  private getWorkingGroup(specId: string): string {
    const workingGroups: { [key: string]: string } = {
      'TS 32.290': 'SA5',
      'TS 32.240': 'SA5',
      'TS 38.331': 'RAN2',
      'TS 33.501': 'SA3',
      'TS 23.501': 'SA2',
      'TS 23.502': 'SA2',
      'TS 29.594': 'SA2',
      'TS 29.122': 'SA2'
    };
    return workingGroups[specId] || 'Unknown';
  }

  private getSpecificationSummary(specId: string): string {
    const summaries: { [key: string]: string } = {
      'TS 32.290': 'Defines the 5G converged charging system using Service Based Interface architecture with CHF as the central charging function.',
      'TS 32.240': 'Establishes the foundational charging architecture and principles for 3GPP networks including online and offline charging.',
      'TS 38.331': 'Specifies the Radio Resource Control protocol for 5G New Radio including connection management and mobility procedures.',
      'TS 33.501': 'Defines the comprehensive security architecture for 5G systems including authentication, authorization, and privacy protection.',
      'TS 23.501': 'Describes the overall system architecture for 5G including network functions, interfaces, and service-based architecture.',
      'TS 23.502': 'Specifies detailed procedures for 5G system operations including registration, session management, and mobility.',
      'TS 29.594': 'Specifies the Network Exposure Function (NEF) Northbound APIs for event exposure and network capability exposure to external applications.',
      'TS 29.122': 'Defines the T8 reference point for Northbound APIs between the Service Capability Exposure Function and external applications.'
    };
    return summaries[specId] || 'Technical specification for 3GPP telecommunications systems.';
  }

  clearCache(): void {
    this.cache.flushAll();
  }

  getCacheStats(): { keys: number; hits: number; misses: number } {
    const stats = this.cache.getStats();
    return {
      keys: this.cache.keys().length,
      hits: stats.hits,
      misses: stats.misses
    };
  }
}