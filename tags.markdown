---
layout: page
title: Tags
permalink: /tags/
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
    <a href="javascript:void(0)" class="tag-link tag-all" onclick="showAllTags()">
      All Tags
    </a>
    {%- for tag in final_tags %}
      {%- assign posts = site.tags[tag] %}
      <a href="javascript:void(0)" class="tag-link" data-tag="{{ tag | slugify }}" onclick="filterByTag('{{ tag | slugify }}')">
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
function filterByTag(tagId) {
  // Hide all sections
  document.querySelectorAll('.tag-section').forEach(section => {
    section.style.display = 'none';
  });
  
  // Show selected section
  const selectedSection = document.getElementById(tagId);
  if (selectedSection) {
    selectedSection.style.display = 'block';
  }
  
  // Update active state
  document.querySelectorAll('.tag-link').forEach(link => {
    link.classList.remove('active');
  });
  const clickedLink = document.querySelector(`.tag-link[data-tag="${tagId}"]`);
  if (clickedLink) {
    clickedLink.classList.add('active');
  }
  
  // Update URL hash without scrolling
  history.pushState(null, null, `#${tagId}`);
  
  // Scroll to section smoothly
  if (selectedSection) {
    selectedSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function showAllTags() {
  // Show all sections
  document.querySelectorAll('.tag-section').forEach(section => {
    section.style.display = 'block';
  });
  
  // Update active state
  document.querySelectorAll('.tag-link').forEach(link => {
    link.classList.remove('active');
  });
  document.querySelector('.tag-all').classList.add('active');
  
  // Clear URL hash
  history.pushState(null, null, window.location.pathname);
}

// Check URL hash on page load and filter accordingly
document.addEventListener('DOMContentLoaded', function() {
  const hash = window.location.hash.substring(1); // Remove the # character
  if (hash) {
    // Check if the tag exists
    const tagSection = document.getElementById(hash);
    if (tagSection) {
      filterByTag(hash);
    } else {
      // If tag doesn't exist, show all tags
      showAllTags();
    }
  } else {
    // No hash, show all tags by default
    showAllTags();
  }
});
</script>
