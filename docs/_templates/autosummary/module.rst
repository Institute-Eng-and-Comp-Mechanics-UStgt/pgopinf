{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

{% if modules %}
.. rubric:: Submodules

.. autosummary::
   :toctree:
   :recursive:

{% for item in modules %}
   {{ item }}
{%- endfor %}
{% endif %}
