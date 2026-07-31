---
layout: page
title: Tags
permalink: /tags/
last_modified_at: 2026-07-30 22:31:19 +0900
---

<div class="tags-page">
  {%- comment -%}
  Sort tags by post count (descending)
  Format: "count|tagname" for sorting, then extract tagname
  {%- endcomment -%}
  {%- assign tag_array = "" | split: "" -%}
  {%- for tag in site.tags %}
    {%- assign count = tag[1].size -%}
    {%- assign padded_count = "0000" | append: count | slice: -4, 4 -%}
    {%- assign tag_info = padded_count | append: "|" | append: tag[0] -%}
    {%- assign tag_array = tag_array | push: tag_info -%}
  {%- endfor %}
  
  {%- assign sorted_tag_array = tag_array | sort | reverse -%}
  {%- assign final_tags = "" | split: "" -%}
  {%- for tag_info in sorted_tag_array %}
    {%- assign parts = tag_info | split: "|" -%}
    {%- assign final_tags = final_tags | push: parts[1] -%}
  {%- endfor %}
  
  <div class="tag-cloud">
    <a href="{{ '/tags/' | relative_url }}" class="tag-link tag-all">
      All Tags
    </a>
    {%- for tag in final_tags %}
      {%- assign posts = site.tags[tag] %}
      <a href="{{ '/tags/' | relative_url }}#{{ tag | slugify }}" class="tag-link" data-tag="{{ tag | slugify }}">
        {{ tag }} <span class="tag-count">({{ posts | size }})</span>
      </a>
    {%- endfor %}
  </div>

  <div class="tag-list">
    {%- for tag in final_tags %}
      {%- assign posts = site.tags[tag] %}
      <section class="tag-section" id="{{ tag | slugify }}" data-tag="{{ tag | slugify }}">
        <h2>{{ tag }}</h2>
        <ul class="post-list">
          {%- for post in posts %}
            <li>
              <span class="post-meta">{{ post.date | date: "%b %-d, %Y" }}</span>
              <h3>
                <a class="post-link" href="{{ post.url | relative_url }}">
                  {{ post.title | escape }}
                </a>
              </h3>
            </li>
          {%- endfor %}
        </ul>
      </section>
    {%- endfor %}
  </div>
</div>

<script>
// Display tags based on current hash (without modifying history)
function displayTags(updateHistory) {
  const hash = window.location.hash.substring(1);
  
  if (hash) {
    const tagSection = document.getElementById(hash);
    if (tagSection) {
      // Hide all sections
      document.querySelectorAll('.tag-section').forEach(section => {
        section.style.display = 'none';
      });
      
      // Show selected section
      tagSection.style.display = 'block';
      
      // Update active state
      document.querySelectorAll('.tag-link').forEach(link => {
        link.classList.remove('active');
      });
      const clickedLink = document.querySelector(`.tag-link[data-tag="${hash}"]`);
      if (clickedLink) {
        clickedLink.classList.add('active');
      }
      
      // Scroll to section smoothly
      tagSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
      return;
    }
  }
  
  // No hash or invalid hash - show all tags
  document.querySelectorAll('.tag-section').forEach(section => {
    section.style.display = 'block';
  });
  
  document.querySelectorAll('.tag-link').forEach(link => {
    link.classList.remove('active');
  });
  document.querySelector('.tag-all').classList.add('active');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
  displayTags(false);
});

// Handle browser back/forward buttons
window.addEventListener('hashchange', function() {
  displayTags(false);
});
</script>
